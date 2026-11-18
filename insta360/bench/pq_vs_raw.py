import io, json, os, sys, tarfile, time
from pathlib import Path
from uuid import UUID
import numpy as np, torch
sys.path.insert(0, '/app/docker/localizer')
import boto3
from botocore.config import Config
import src.localize as L
from src.map import load_map
from core.calibration import load_global_calibration
from core.camera_config import PinholeCameraConfig
from core.axis_convention import AxisConvention
from core.image_preprocess import canonicalize_image
from neural_networks.models import load_aliked
from core.model_wrappers import make_local_feature_extractor

RID, CAPTURE_TAR, QUERIES = sys.argv[1], sys.argv[2], Path(sys.argv[3])
s3 = boto3.client('s3', endpoint_url=os.environ['MINIO_ENDPOINT_URL'], aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                  aws_secret_access_key=os.environ['MINIO_SECRET_KEY'], config=Config(signature_version='s3v4'))
L.load_models()
cal = load_global_calibration(Path('/etc/placeframe/calibration/global.json'), os.environ['LOCALIZER_SHA'])
class _Metrics:  # the build's recorded metrics, as the API passes them to load_map
    def model_dump(self): return json.loads(Path('/prof/metrics.json').read_text())
m = load_map(UUID(RID), s3, os.environ['RECONSTRUCTIONS_BUCKET'], Path('/tmp/maps'), _Metrics())
print('map loaded:', len(m.ordered_image_ids), 'images', flush=True)
tar = tarfile.open(CAPTURE_TAR)
manifest = json.load(tar.extractfile('manifest.json'))
orient = {f"{r['id']}/{c['id']}": c['camera_config']['orientation'] for r in manifest['rigs'] for c in r['cameras']}
aliked = load_aliked(device=L.DEVICE); aliked.dkd.n_limit = 2500  # reconstruction default (max_keypoints_per_image)
extract = make_local_feature_extractor(aliked)
raw_cache, kp_check = {}, []
def raw_descriptors(image_id):
    if image_id not in raw_cache:
        name = m.images[image_id].name
        img = canonicalize_image(tar.extractfile(name).read(), orient['/'.join(name.split('/')[:2])])
        t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).div(255.0).unsqueeze(0).to(L.DEVICE)
        with torch.inference_mode():
            k, d = extract(t)
        k = k.cpu().numpy(); d = d.cpu().numpy().astype(np.float32)
        stored = m.keypoints[image_id]
        kp_check.append(k.shape == stored.shape and float(np.abs(k - stored).max()) < 0.5)
        raw_cache[image_id] = d
    return raw_cache[image_id]
pq_decode = L.decode_descriptors
mode = {'raw': False}
def patched(opq, pq, codes):
    return {i: raw_descriptors(i) for i in codes} if mode['raw'] else pq_decode(opq, pq, codes)
L.decode_descriptors = patched
rows = [json.loads(l) for l in QUERIES.read_text().splitlines()]
out = {}
for name in ('pq', 'raw'):
    mode['raw'] = name == 'raw'; ok = 0; inl = []; conf = []
    for r in rows:
        cam = PinholeCameraConfig.model_validate(r['camera_config'])
        try:
            _, met = L.localize_image_against_reconstruction(m, cam, AxisConvention(r['axis_convention']), Path(r['image']).read_bytes(),
                                                             None, None, os.environ['LOCALIZER_SHA'], cal)
            ok += 1; inl.append(met.num_inliers if hasattr(met, 'num_inliers') else 0); conf.append(met.confidence_loose)
        except Exception as e:
            pass
    out[name] = {'localized': ok, 'of': len(rows), 'median_inliers': float(np.median(inl)) if inl else None,
                 'median_confidence': round(float(np.median(conf)), 3) if conf else None}
    print(name, out[name], flush=True)
print('keypoints identical to the stored map keypoints:', sum(kp_check), 'of', len(kp_check), 'images')
