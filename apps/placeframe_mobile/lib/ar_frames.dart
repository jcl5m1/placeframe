import 'dart:async';

import 'package:flutter/services.dart';

import 'geo.dart';

// One camera frame from the platform's AR tracking (ARKit on iOS, ARCore on Android), no AR view involved.
class ArFrame {
  ArFrame(this.jpeg, this.width, this.height, this.fx, this.fy, this.cx, this.cy, this.worldFromCamera, this.timestamp,
      this.tracking);

  factory ArFrame.fromMap(Map<dynamic, dynamic> m) {
    final pose = (m['pose'] as List<dynamic>).map((v) => (v as num).toDouble()).toList();
    double d(String k) => (m[k] as num).toDouble();
    return ArFrame(m['jpeg'] as Uint8List, (m['width'] as num).toInt(), (m['height'] as num).toInt(), d('fx'), d('fy'),
        d('cx'), d('cy'), Pose.fromColumnMajor(pose).withOpenCvCameraAxes(), d('timestamp'), m['tracking'] as String);
  }

  final Uint8List jpeg; // the camera image as the sensor delivers it (not rotated to the screen)
  final int width, height;
  final double fx, fy, cx, cy; // intrinsics of this image, pixels
  final Pose worldFromCamera; // AR world (y up, metres) from camera, OpenCV camera axes
  final double timestamp; // seconds, platform clock
  final String tracking; // "normal" | "limited" | "not_available"

  bool get trackingOk => tracking == 'normal';

  // Placeframe camera_config for this frame (the image is sent unrotated: orientation TOP_LEFT).
  Map<String, dynamic> get cameraConfig => {
        'width': width,
        'height': height,
        'orientation': 'TOP_LEFT',
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
      };
}

// Platform channel to the native AR session. Frames arrive at most every [intervalSeconds].
class ArFrames {
  static const _methods = MethodChannel('placeframe/ar');
  static const _events = EventChannel('placeframe/ar/frames');

  static Stream<ArFrame>? _stream;

  static Future<void> start({required double intervalSeconds, int jpegQuality = 90}) =>
      _methods.invokeMethod('start', {'interval': intervalSeconds, 'quality': jpegQuality});

  static Future<void> stop() => _methods.invokeMethod('stop');

  static Future<bool> available() async => (await _methods.invokeMethod<bool>('available')) ?? false;

  static Stream<ArFrame> get frames =>
      _stream ??= _events.receiveBroadcastStream().map((e) => ArFrame.fromMap(e as Map<dynamic, dynamic>));
}
