import 'package:shared_preferences/shared_preferences.dart';

// Server settings, persisted on the device. The defaults match a Placeframe stack behind one public HTTPS host
// (API at the root, Keycloak under /auth), as Placeframe's gateway serves it.
class Config {
  Config._(this._prefs);

  final SharedPreferences _prefs;

  static Future<Config> load() async => Config._(await SharedPreferences.getInstance());

  String get serverUrl => _prefs.getString('serverUrl') ?? '';
  set serverUrl(String v) => _prefs.setString('serverUrl', v.trim().replaceAll(RegExp(r'/+$'), ''));

  String get realm => _prefs.getString('realm') ?? 'placeframe-dev';
  set realm(String v) => _prefs.setString('realm', v.trim());

  String get clientId => _prefs.getString('clientId') ?? 'placeframe-api';
  set clientId(String v) => _prefs.setString('clientId', v.trim());

  String get username => _prefs.getString('username') ?? '';
  set username(String v) => _prefs.setString('username', v.trim());

  double get mapRadiusMeters => _prefs.getDouble('mapRadius') ?? 100;
  set mapRadiusMeters(double v) => _prefs.setDouble('mapRadius', v);

  double get localizeIntervalSeconds => _prefs.getDouble('localizeInterval') ?? 1.0;
  set localizeIntervalSeconds(double v) => _prefs.setDouble('localizeInterval', v);

  double get captureIntervalSeconds => _prefs.getDouble('captureInterval') ?? 0.5;
  set captureIntervalSeconds(double v) => _prefs.setDouble('captureInterval', v);

  String get tokenUrl => '$serverUrl/auth/realms/$realm/protocol/openid-connect/token';
}
