import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:geolocator/geolocator.dart';
import 'package:material_ui/material_ui.dart';
import 'package:permission_handler/permission_handler.dart';

import 'api.dart';
import 'ar_frames.dart';
import 'auth.dart';
import 'capture.dart';
import 'config.dart';
import 'geo.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final config = await Config.load();
  final auth = Auth(config);
  runApp(PlaceframeApp(config: config, auth: auth, api: PlaceframeApi(config, auth)));
}

class PlaceframeApp extends StatefulWidget {
  const PlaceframeApp({super.key, required this.config, required this.auth, required this.api});
  final Config config;
  final Auth auth;
  final PlaceframeApi api;

  @override
  State<PlaceframeApp> createState() => _PlaceframeAppState();
}

class _PlaceframeAppState extends State<PlaceframeApp> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Placeframe',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF4CC2FF), brightness: Brightness.dark)),
      home: !widget.auth.loggedIn
          ? LoginScreen(config: widget.config, auth: widget.auth, onLoggedIn: () => setState(() {}))
          : Scaffold(
              appBar: AppBar(title: Text(['Localize', 'Capture', 'Settings'][_tab])),
              body: IndexedStack(index: _tab, children: [
                LocalizeScreen(config: widget.config, api: widget.api),
                CaptureScreen(config: widget.config, api: widget.api),
                SettingsScreen(config: widget.config, auth: widget.auth, onLogout: () => setState(() {})),
              ]),
              bottomNavigationBar: NavigationBar(
                selectedIndex: _tab,
                onDestinationSelected: (i) => setState(() => _tab = i),
                destinations: const [
                  NavigationDestination(icon: Icon(Icons.my_location), label: 'Localize'),
                  NavigationDestination(icon: Icon(Icons.videocam), label: 'Capture'),
                  NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
                ],
              ),
            ),
    );
  }
}

// ------------------------------------------------------------------------------------------------ login

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key, required this.config, required this.auth, required this.onLoggedIn});
  final Config config;
  final Auth auth;
  final VoidCallback onLoggedIn;

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  late final _server = TextEditingController(text: widget.config.serverUrl);
  late final _user = TextEditingController(text: widget.config.username);
  final _password = TextEditingController();
  String? _error;
  bool _busy = false;

  Future<void> _login() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    widget.config.serverUrl = _server.text;
    try {
      await widget.auth.login(_user.text, _password.text);
      widget.onLoggedIn();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Placeframe')),
        body: ListView(padding: const EdgeInsets.all(20), children: [
          TextField(
              controller: _server,
              keyboardType: TextInputType.url,
              decoration: const InputDecoration(labelText: 'Server URL', hintText: 'https://your-placeframe-host')),
          TextField(controller: _user, decoration: const InputDecoration(labelText: 'Username')),
          TextField(
              controller: _password,
              obscureText: true,
              decoration: const InputDecoration(labelText: 'Password'),
              onSubmitted: (_) => _login()),
          const SizedBox(height: 16),
          FilledButton(onPressed: _busy ? null : _login, child: Text(_busy ? 'Logging in…' : 'Log in')),
          if (_error != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text(_error!, style: const TextStyle(color: Colors.redAccent))),
        ]),
      );
}

// ------------------------------------------------------------------------------------------------ localize

class LocalizeScreen extends StatefulWidget {
  const LocalizeScreen({super.key, required this.config, required this.api});
  final Config config;
  final PlaceframeApi api;

  @override
  State<LocalizeScreen> createState() => _LocalizeScreenState();
}

class _LocalizeScreenState extends State<LocalizeScreen> {
  List<MapInfo> _maps = [];
  final Set<String> _selected = {};
  bool _nearby = true, _loadingMaps = false, _running = false, _inFlight = false;
  Position? _fix;
  String? _status;
  Localization? _last;
  ArFrame? _lastFrame;
  int _attempts = 0, _successes = 0;
  StreamSubscription<ArFrame>? _sub;

  @override
  void initState() {
    super.initState();
    _loadMaps();
  }

  @override
  void dispose() {
    _stop();
    super.dispose();
  }

  Future<void> _loadMaps() async {
    setState(() {
      _loadingMaps = true;
      _status = null;
    });
    try {
      if (_nearby) {
        _fix = await currentFix();
        _maps = _fix == null
            ? []
            : await widget.api.maps(latitude: _fix!.latitude, longitude: _fix!.longitude, altitude: _fix!.altitude);
        if (_fix == null) _status = 'No location: allow location access, or show all maps';
      } else {
        _maps = await widget.api.maps();
      }
      _maps.sort((a, b) => (b.active ? 1 : 0) - (a.active ? 1 : 0));
      _selected.retainAll(_maps.map((m) => m.id));
      if (_selected.isEmpty) _selected.addAll(_maps.where((m) => m.active).map((m) => m.id).take(_nearby ? 4 : 1));
    } catch (e) {
      _status = '$e';
    } finally {
      if (mounted) setState(() => _loadingMaps = false);
    }
  }

  Future<void> _start() async {
    if (_selected.isEmpty) return setState(() => _status = 'Select at least one map');
    if (!await cameraAllowed()) return setState(() => _status = 'Camera permission is needed');
    _attempts = _successes = 0;
    try {
      await ArFrames.start(intervalSeconds: widget.config.localizeIntervalSeconds);
    } catch (e) {
      return setState(() => _status = 'AR tracking unavailable: $e');
    }
    _sub = ArFrames.frames.listen(_onFrame);
    setState(() {
      _running = true;
      _status = 'Point the camera at the mapped space…';
    });
  }

  Future<void> _stop() async {
    await _sub?.cancel();
    _sub = null;
    if (_running) await ArFrames.stop();
    if (mounted) setState(() => _running = false);
  }

  Future<void> _onFrame(ArFrame f) async {
    _lastFrame = f;
    if (_inFlight) return; // one request at a time; later frames are newer anyway
    _inFlight = true;
    _attempts++;
    try {
      final results = await widget.api.localize(f.jpeg, f.cameraConfig, _selected.toList());
      if (results.isEmpty) {
        _status = 'Not localized';
      } else {
        results.sort((a, b) => b.confidence.compareTo(a.confidence));
        _last = results.first;
        _successes++;
        _status = 'Localized';
      }
    } catch (e) {
      _status = e is ApiException && e.status == 404 ? 'Not localized' : '$e';
    } finally {
      _inFlight = false;
      if (mounted) setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    final last = _last;
    final mapName = last == null ? null : _maps.where((m) => m.id == last.mapId).map((m) => m.name).firstOrNull;
    return ListView(padding: const EdgeInsets.all(16), children: [
      Row(children: [
        Expanded(
          child: SegmentedButton<bool>(
            segments: const [ButtonSegment(value: true, label: Text('Nearby')), ButtonSegment(value: false, label: Text('All maps'))],
            selected: {_nearby},
            onSelectionChanged: _running ? null : (v) => setState(() => _nearby = v.first),
          ),
        ),
        IconButton(onPressed: _loadingMaps || _running ? null : _loadMaps, icon: const Icon(Icons.refresh)),
      ]),
      if (_nearby && _fix != null)
        Text('GPS ${_fix!.latitude.toStringAsFixed(5)}, ${_fix!.longitude.toStringAsFixed(5)} '
            '± ${_fix!.accuracy.round()} m · radius ${widget.config.mapRadiusMeters.round()} m'),
      if (_loadingMaps) const LinearProgressIndicator(),
      if (!_loadingMaps && _maps.isEmpty)
        const Padding(padding: EdgeInsets.all(8), child: Text('No maps found. Try “All maps”: maps without a GPS placement are not found by location.')),
      for (final m in _maps)
        CheckboxListTile(
          dense: true,
          value: _selected.contains(m.id),
          onChanged: _running ? null : (v) => setState(() => v == true ? _selected.add(m.id) : _selected.remove(m.id)),
          title: Text(m.name),
          subtitle: Text(m.active ? 'active' : 'inactive'),
        ),
      const SizedBox(height: 8),
      FilledButton.icon(
        onPressed: _running ? _stop : _start,
        icon: Icon(_running ? Icons.stop : Icons.play_arrow),
        label: Text(_running ? 'Stop localizing' : 'Start localizing'),
      ),
      const SizedBox(height: 12),
      if (_status != null) Text(_status!, style: Theme.of(context).textTheme.titleMedium),
      if (_lastFrame != null) Text('Tracking: ${_lastFrame!.tracking} · ${_lastFrame!.width}×${_lastFrame!.height}'),
      if (_attempts > 0) Text('Localized $_successes of $_attempts frames'),
      if (last != null)
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Last localization: ${mapName ?? last.mapId.substring(0, 8)}', style: Theme.of(context).textTheme.titleSmall),
              Text('Position in map: ${last.mapFromCamera.t.map((v) => v.toStringAsFixed(2)).join(', ')} m'),
              Text('Confidence ${last.confidence.toStringAsFixed(2)} · ${last.latencyMs} ms'),
              if (last.metrics['num_inliers'] != null) Text('Inliers: ${last.metrics['num_inliers']}'),
            ]),
          ),
        ),
    ]);
  }
}

// ------------------------------------------------------------------------------------------------ capture

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key, required this.config, required this.api});
  final Config config;
  final PlaceframeApi api;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

// A capture kept on the device, with its server-side progress (saved next to the capture as `<id>.json`).
class LocalCapture {
  LocalCapture(this.json);
  final Map<String, dynamic> json;
  String get id => json['id'] as String;
  String get name => json['name'] as String;
  String? get status => json['status'] as String?;

  static Future<List<LocalCapture>> all() async {
    final root = await CaptureSession.root();
    final files = root.listSync().whereType<File>().where((f) => f.path.endsWith('.json')).toList();
    final list = [for (final f in files) LocalCapture(jsonDecode(await f.readAsString()) as Map<String, dynamic>)];
    list.sort((a, b) => (b.json['recorded_at'] as String).compareTo(a.json['recorded_at'] as String));
    return list;
  }

  Future<void> save() async =>
      File('${(await CaptureSession.root()).path}/$id.json').writeAsString(jsonEncode(json));
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _name = TextEditingController();
  CaptureSession? _session;
  StreamSubscription<ArFrame>? _frames;
  StreamSubscription<Position>? _gps;
  Position? _fix;
  String _tracking = '-';
  DateTime? _startedAt;
  List<LocalCapture> _captures = [];
  bool _usePriors = false;
  String? _message;
  final Set<String> _busy = {};
  Timer? _ticker;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  Future<void> _reload() async {
    _captures = await LocalCapture.all();
    if (mounted) setState(() {});
  }

  Future<void> _record() async {
    if (!await cameraAllowed()) return setState(() => _message = 'Camera permission is needed');
    final session = await CaptureSession.start();
    try {
      await ArFrames.start(intervalSeconds: widget.config.captureIntervalSeconds);
    } catch (e) {
      return setState(() => _message = 'AR tracking unavailable: $e');
    }
    _frames = ArFrames.frames.listen((f) {
      _tracking = f.tracking;
      session.addFrame(f);
    });
    if (await ensureLocationPermission()) {
      _gps = Geolocator.getPositionStream(locationSettings: const LocationSettings(accuracy: LocationAccuracy.best))
          .listen((p) {
        _fix = p;
        session.addFix(p);
      });
    }
    _ticker = Timer.periodic(const Duration(milliseconds: 500), (_) => setState(() {}));
    setState(() {
      _session = session;
      _startedAt = DateTime.now();
      _message = null;
    });
  }

  Future<void> _stopRecording() async {
    final session = _session!;
    _ticker?.cancel();
    await _frames?.cancel();
    await _gps?.cancel();
    await ArFrames.stop();
    setState(() => _session = null);
    try {
      final tar = await session.finish(widget.config.captureIntervalSeconds);
      final name = _name.text.trim().isEmpty ? 'phone capture ${session.id}' : _name.text.trim();
      await LocalCapture({
        'id': session.id,
        'name': name,
        'recorded_at': _startedAt!.toUtc().toIso8601String(),
        'frames': session.frames,
        'tar': tar.path,
        'tar_bytes': await tar.length(),
        'has_location': await File('${session.dir.path}/location.json').exists(),
        'status': 'recorded',
      }).save();
      await session.dir.delete(recursive: true); // the tar holds everything
      _message = 'Recorded ${session.frames} frames';
    } catch (e) {
      _message = '$e';
    }
    await _reload();
  }

  Future<void> _upload(LocalCapture c) async {
    setState(() => _busy.add(c.id));
    try {
      c.json['status'] = 'uploading';
      setState(() {});
      final captureId = await widget.api
          .uploadCapture(File(c.json['tar'] as String), c.name, DateTime.parse(c.json['recorded_at'] as String));
      c.json['capture_session_id'] = captureId;
      final options = _usePriors ? {'use_prior_position': true, 'pose_prior_position_sigma_m': 2.0} : <String, dynamic>{};
      final rec = await widget.api.createReconstruction(captureId, options);
      c.json['reconstruction_id'] = rec['id'];
      c.json['status'] = rec['status'];
    } catch (e) {
      c.json['status'] = 'upload failed: $e';
    } finally {
      await c.save();
      _busy.remove(c.id);
      await _reload();
    }
  }

  Future<void> _refresh(LocalCapture c) async {
    final rid = c.json['reconstruction_id'] as String?;
    if (rid == null) return;
    try {
      final rec = await widget.api.reconstruction(rid);
      final cur = rec['progress_current'], tot = rec['progress_total'];
      c.json['status'] = '${rec['status']}${cur != null && tot != null ? ' $cur/$tot' : ''}';
      if (rec['status'] == 'failed') c.json['status'] = 'failed: ${rec['error'] ?? ''}';
      await c.save();
    } catch (e) {
      _message = '$e';
    }
    await _reload();
  }

  Future<void> _createMap(LocalCapture c) async {
    try {
      final map = await widget.api.createMap(c.json['reconstruction_id'] as String, c.name);
      c.json['map_id'] = map.id;
      c.json['status'] = 'map created';
      await c.save();
    } catch (e) {
      _message = '$e';
    }
    await _reload();
  }

  @override
  Widget build(BuildContext context) {
    final s = _session;
    return ListView(padding: const EdgeInsets.all(16), children: [
      TextField(controller: _name, enabled: s == null, decoration: const InputDecoration(labelText: 'Capture name (optional)')),
      SwitchListTile(
        contentPadding: EdgeInsets.zero,
        value: _usePriors,
        onChanged: (v) => setState(() => _usePriors = v),
        title: const Text('Use AR poses as position priors'),
        subtitle: const Text('Guides the build with the phone’s tracking (2 m prior). Off = Placeframe defaults.'),
      ),
      FilledButton.icon(
        onPressed: s == null ? _record : _stopRecording,
        icon: Icon(s == null ? Icons.fiber_manual_record : Icons.stop),
        label: Text(s == null ? 'Start recording' : 'Stop recording'),
      ),
      if (s != null) ...[
        const SizedBox(height: 8),
        Text('${s.frames} frames · ${DateTime.now().difference(_startedAt!).inSeconds} s · tracking $_tracking'),
        Text(_fix == null ? 'GPS: waiting for a fix…' : 'GPS ± ${_fix!.accuracy.round()} m'),
        const Text('Walk slowly; keep the phone steady and the scene textured.'),
      ],
      if (_message != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_message!)),
      const Divider(height: 32),
      Row(children: [
        Text('Captures on this device', style: Theme.of(context).textTheme.titleSmall),
        const Spacer(),
        IconButton(
            onPressed: () async {
              for (final c in _captures) {
                await _refresh(c);
              }
            },
            icon: const Icon(Icons.refresh)),
      ]),
      for (final c in _captures)
        Card(
          child: ListTile(
            title: Text(c.name),
            subtitle: Text('${c.json['frames']} frames · ${((c.json['tar_bytes'] as num) / 1e6).toStringAsFixed(1)} MB'
                '${c.json['has_location'] == true ? ' · GPS' : ' · no GPS'}\n${c.status ?? ''}'),
            isThreeLine: true,
            trailing: _busy.contains(c.id)
                ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                : PopupMenuButton<String>(
                    onSelected: (a) async {
                      if (a == 'upload') await _upload(c);
                      if (a == 'refresh') await _refresh(c);
                      if (a == 'map') await _createMap(c);
                      if (a == 'delete') {
                        await CaptureSession.delete(c.id);
                        await _reload();
                      }
                    },
                    itemBuilder: (_) => [
                      if (c.json['capture_session_id'] == null) const PopupMenuItem(value: 'upload', child: Text('Upload and build')),
                      if (c.json['reconstruction_id'] != null) const PopupMenuItem(value: 'refresh', child: Text('Refresh status')),
                      if ((c.status ?? '').startsWith('succeeded') && c.json['map_id'] == null)
                        const PopupMenuItem(value: 'map', child: Text('Create map')),
                      const PopupMenuItem(value: 'delete', child: Text('Delete from device')),
                    ],
                  ),
          ),
        ),
    ]);
  }
}

// ------------------------------------------------------------------------------------------------ settings

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.config, required this.auth, required this.onLogout});
  final Config config;
  final Auth auth;
  final VoidCallback onLogout;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  @override
  Widget build(BuildContext context) {
    final c = widget.config;
    Widget number(String label, double value, void Function(double) set) => TextFormField(
          initialValue: value.toString(),
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          decoration: InputDecoration(labelText: label),
          onChanged: (v) => double.tryParse(v) != null ? set(double.parse(v)) : null,
        );
    return ListView(padding: const EdgeInsets.all(16), children: [
      Text('Server: ${c.serverUrl}\nUser: ${c.username}'),
      TextFormField(initialValue: c.realm, decoration: const InputDecoration(labelText: 'Keycloak realm'), onChanged: (v) => c.realm = v),
      TextFormField(initialValue: c.clientId, decoration: const InputDecoration(labelText: 'Client id'), onChanged: (v) => c.clientId = v),
      number('Nearby map radius (m)', c.mapRadiusMeters, (v) => c.mapRadiusMeters = v),
      number('Seconds between localization frames', c.localizeIntervalSeconds, (v) => c.localizeIntervalSeconds = v),
      number('Seconds between capture frames', c.captureIntervalSeconds, (v) => c.captureIntervalSeconds = v),
      const SizedBox(height: 16),
      OutlinedButton(
          onPressed: () {
            widget.auth.logout();
            widget.onLogout();
          },
          child: const Text('Log out')),
    ]);
  }
}

// ------------------------------------------------------------------------------------------------ location

// Android needs the runtime camera permission before ARCore opens the camera; on iOS ARKit asks for it itself.
Future<bool> cameraAllowed() async => !Platform.isAndroid || await Permission.camera.request().isGranted;

Future<bool> ensureLocationPermission() async {
  if (!await Geolocator.isLocationServiceEnabled()) return false;
  var p = await Geolocator.checkPermission();
  if (p == LocationPermission.denied) p = await Geolocator.requestPermission();
  return p == LocationPermission.always || p == LocationPermission.whileInUse;
}

Future<Position?> currentFix() async {
  if (!await ensureLocationPermission()) return null;
  try {
    return await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.best, timeLimit: Duration(seconds: 15)));
  } catch (_) {
    return await Geolocator.getLastKnownPosition();
  }
}

// ECEF of a GPS fix (exposed for callers that query maps by position).
List<double> ecefOf(Position p) => ecefFromGeodetic(p.latitude, p.longitude, p.altitude);
