# GPS: putting maps on the globe

Placeframe can find the maps near a device: every map has a placement (a transform from the map's coordinates to
ECEF, Earth-centred metres), and `GET /localization-maps?position_x=&position_y=&position_z=&radius=` returns the
maps with a camera inside that sphere. Maps without a placement sit at ECEF (0,0,0), the centre of the Earth, so a
location-based lookup never finds them. This page covers getting a GPS fix into every capture so maps are placed
automatically, and placing existing maps by hand.

## How it flows

1. **Capture**: the capture carries the device's location as `location.json` at the root of the capture tar:
   `{"latitude": deg, "longitude": deg, "altitude": m, "horizontal_accuracy": m, "map_up_axis": "-y", ...}`
   (only latitude and longitude are required).
2. **Reconstructor**: stores it with the build (`<reconstruction id>/location.json` in the reconstructions bucket).
   For Unity captures it adds `map_up_axis: "-y"` (Unity y-up becomes -y in the map's OpenCV frame).
3. **API**: when a localization map is created *without* a placement (identity transform), it is placed from that
   location: the centroid of the map's cameras sits at the fix, the map's up axis along local up. Fixes worse than
   200 m horizontal accuracy are ignored. Maps created with a placement are left alone.
4. **Merges** into an existing map (the mapper web page, `add_recording_to_map.py`) copy the base map's placement,
   so placing a map once carries through every later merge.

What one fix does **not** give: the map's heading (rotation about vertical). That does not matter for selecting
maps by distance; for AR content that must line up with the real world, refine heading in the MapRegistrationTool.

## Insta360 X5

The X5 has **no GPS receiver**. It records location only from a connected controller
([Insta360 X5 manual: GPS data](https://onlinemanual.insta360.com/x5/en-us/operating_tutorials/clips/gps)):

1. Connect the X5 to the **Insta360 phone app** (Bluetooth/Wi-Fi), with location permission granted to the app.
2. On the app's **Shooting page, turn on the GPS switch** before you start recording.
3. Start and stop recording **from the app** (or from an Insta360 **GPS remote**, which also supplies GPS).
   Recording with the camera's own buttons stores **no** GPS.
4. Keep the phone next to the camera and the app in the foreground while recording.
5. Upload the **original `.insv`** (exports such as MP4 drop the GPS).

GPS Activity Stats in the app is a different feature: it does **not** put GPS into the original file.

Check a recording: `python gps_placement.py insv-gps VID_xxx.insv` prints the number of fixes and their median.
On upload, a first-time build gets `location.json` from `vio_to_capture.py`; a merge logs the recording's GPS in
the `geolocate` stage and places the new map only if the base map has no placement yet.

> Untested on a real GPS-carrying `.insv`: the field names are read defensively (`lat`/`latitude`, `lon`/`lng`/
> `longitude`, `altitude`/`alt`). If `insv-gps` reports 0 fixes on a recording shot with the app's GPS switch on,
> that parser is the place to look.

## Phone capture tool (Android)

`packages/unity/Placeframe/.../Core/Runtime/CaptureManager.cs` starts Unity's location service when a capture
starts (asking for the location permission on Android), keeps the most accurate fix seen while recording, and
writes `location.json` into the capture folder when it stops. The capture tool's Android manifest declares
`ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION`.

- Unity's `Input.location` uses Android's GPS and network providers (not Google Play Services' fused provider);
  outdoors expect a few metres, indoors 10–50 m or no fix at all. Starting a capture near a window or doorway helps.
- Android reports altitude above the WGS84 ellipsoid; iOS above mean sea level (tens of metres apart). Either is
  close enough for selecting maps within ~100 m.
- No fix (permission denied, location off, no signal in 20 s): no `location.json`; the map stays unplaced.

> Not compiled here (no Unity editor on the build host). Compile with the command in `apps/AndroidMobile/CLAUDE.md`
> before building the app.

## Clients selecting maps by location

`VisualPositioningSystem.SetLocalizationMaps(ecefPosition, radius)` (Unity package) already loads the maps near a
position. The capture tool selects its validation map by hand. MakeItSing has the location → ECEF → lookup path but
deliberately looks up ECEF (0,0,0) (see `apps/MakeItSing/CLAUDE.md`): switching it to the device location is a
one-line change once maps are placed, to be agreed with that app's author.

## Placing existing maps by hand

```bash
python gps_placement.py place --map-id <id> --lat 37.4219 --lon -122.0840 [--alt 10] [--heading 0] [--dry-run]
python gps_placement.py check --lat 37.4219 --lon -122.0840 --radius 100   # what a client there would load
```

The MapRegistrationTool (Unity desktop app, `apps/MapRegistrationTool`) is Placeframe's interactive placement UI:
it uses a Cesium globe scene with a latitude/longitude/height inspector, for fine placement and heading.
