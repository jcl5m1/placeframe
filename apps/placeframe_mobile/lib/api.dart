import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'auth.dart';
import 'config.dart';
import 'geo.dart';

class MapInfo {
  MapInfo(this.json);
  final Map<String, dynamic> json;
  String get id => json['id'] as String;
  String get name => (json['name'] as String?) ?? id.substring(0, 8);
  bool get active => json['active'] == true;
  String get reconstructionId => json['reconstruction_id'] as String;
}

class Localization {
  Localization(this.mapId, this.mapFromCamera, this.confidence, this.latencyMs, this.metrics);
  final String mapId;
  final Pose mapFromCamera; // OpenCV camera axes (x right, y down, z forward) in the map's frame
  final double confidence;
  final int latencyMs;
  final Map<String, dynamic> metrics;
}

// The Placeframe API calls this app uses (same endpoints as the Unity client and the Python tools).
class PlaceframeApi {
  PlaceframeApi(this.config, this.auth);

  final Config config;
  final Auth auth;
  final http.Client _client = http.Client();

  Future<Map<String, String>> _headers() async => {'Authorization': 'Bearer ${await auth.token()}'};

  Uri _uri(String path, [Map<String, String>? query]) => Uri.parse('${config.serverUrl}$path').replace(queryParameters: query);

  Future<dynamic> _get(String path, [Map<String, String>? query]) async {
    final r = await _client.get(_uri(path, query), headers: await _headers());
    return _check(r);
  }

  Future<dynamic> _postJson(String path, Map<String, dynamic> body) async {
    final r = await _client.post(_uri(path),
        headers: {...await _headers(), 'Content-Type': 'application/json'}, body: jsonEncode(body));
    return _check(r);
  }

  dynamic _check(http.Response r) {
    if (r.statusCode >= 400) throw ApiException(r.statusCode, r.body);
    return r.body.isEmpty ? null : jsonDecode(r.body);
  }

  // All maps, or (with a location) the maps with a camera within [radiusMeters] of it.
  Future<List<MapInfo>> maps({double? latitude, double? longitude, double altitude = 0, double? radiusMeters}) async {
    Map<String, String>? query;
    if (latitude != null && longitude != null) {
      final p = ecefFromGeodetic(latitude, longitude, altitude);
      query = {
        'position_x': '${p[0]}',
        'position_y': '${p[1]}',
        'position_z': '${p[2]}',
        'radius': '${radiusMeters ?? config.mapRadiusMeters}',
      };
    }
    final list = await _get('/localization-maps', query) as List<dynamic>;
    return list.map((m) => MapInfo(m as Map<String, dynamic>)).toList();
  }

  Future<List<Localization>> localize(Uint8List jpeg, Map<String, dynamic> cameraConfig, List<String> mapIds) async {
    final t = Stopwatch()..start();
    final req = http.MultipartRequest('POST', _uri('/localize'))
      ..headers.addAll(await _headers())
      ..fields['map_ids'] = jsonEncode(mapIds)
      ..fields['camera_config'] = jsonEncode(cameraConfig)
      ..fields['axis_convention'] = 'OPENCV'
      ..files.add(http.MultipartFile.fromBytes('image', jpeg, filename: 'frame.jpg'));
    final r = await http.Response.fromStream(await _client.send(req));
    final latency = t.elapsedMilliseconds;
    if (r.statusCode >= 400) throw ApiException(r.statusCode, r.body);
    final list = (jsonDecode(r.body) as List<dynamic>).cast<Map<String, dynamic>>();
    return list.map((m) {
      final tr = m['camera_from_map_transform'] as Map<String, dynamic>;
      final metrics = (m['metrics'] as Map<String, dynamic>?) ?? {};
      return Localization(m['id'] as String, Pose.fromTransformJson(tr).inverse(),
          ((metrics['confidence_loose'] as num?) ?? 0).toDouble(), latency, metrics);
    }).toList();
  }

  // Uploads a capture tar; returns the capture session id.
  Future<String> uploadCapture(File tar, String name, DateTime recordedAt) async {
    final req = http.MultipartRequest('POST', _uri('/capture_sessions'))
      ..headers.addAll(await _headers())
      ..fields['device_type'] = 'ARFoundation'
      ..fields['name'] = name
      ..fields['recorded_at'] = recordedAt.toUtc().toIso8601String()
      ..files.add(await http.MultipartFile.fromPath('data', tar.path, filename: 'capture.tar'));
    final r = await http.Response.fromStream(await _client.send(req));
    return (_check(r) as Map<String, dynamic>)['id'] as String;
  }

  Future<Map<String, dynamic>> createReconstruction(String captureId, Map<String, dynamic> options) async =>
      await _postJson('/reconstructions', {
        'create': {'capture_session_id': captureId},
        'options': options,
      }) as Map<String, dynamic>;

  Future<Map<String, dynamic>> reconstruction(String id) async =>
      await _get('/reconstructions/$id') as Map<String, dynamic>;

  // Creates a map from a finished reconstruction. No placement is sent: the server places it from the capture's
  // location.json when the capture carried one.
  Future<MapInfo> createMap(String reconstructionId, String name) async => MapInfo(await _postJson('/localization-maps', {
        'reconstruction_id': reconstructionId,
        'name': name,
        'position_x': 0.0, 'position_y': 0.0, 'position_z': 0.0,
        'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation_z': 0.0, 'rotation_w': 1.0,
        'color': 0,
        'active': true,
      }) as Map<String, dynamic>);
}

class ApiException implements Exception {
  ApiException(this.status, this.body);
  final int status;
  final String body;
  @override
  String toString() {
    try {
      final detail = (jsonDecode(body) as Map<String, dynamic>)['detail'];
      if (detail != null) return 'HTTP $status: $detail';
    } catch (_) {}
    return 'HTTP $status';
  }
}
