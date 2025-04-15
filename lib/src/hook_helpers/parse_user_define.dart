import 'dart:io';
import 'package:yaml/yaml.dart';

/// Parse the user defined exclude modules from pubspec.yaml
///
/// Returns a list of excluded module names
Map<String, Object> parseUserDefinedOptions(String pubspecPath) {
  try {
    // Read the pubspec.yaml file
    final File file = File(pubspecPath);
    if (!file.existsSync()) {
      return {};
    }

    // Parse the YAML content
    final String yamlContent = file.readAsStringSync();
    final dynamic yamlMap = loadYaml(yamlContent);

    final defs = <String, Object>{
      "defines": {
        "common": <String, String>{},
        "android": <String, String>{},
        "ios": <String, String>{},
        "linux": <String, String>{},
        "windows": <String, String>{},
        "macos": <String, String>{},
        "web": <String, String>{},
      },
      "options": <String, Object>{},
    };
    // Navigate to the hooks.user_defines.mnn.exclude_modules section
    if (yamlMap is YamlMap &&
        yamlMap['hooks'] is YamlMap &&
        yamlMap['hooks']['user_defines'] is YamlMap &&
        yamlMap['hooks']['user_defines']['mnn'] is YamlMap) {
      if (yamlMap['hooks']['user_defines']['mnn'] case {"defines": final YamlMap defines}) {
        if (defines case {"common": final YamlMap common}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["common"]! as Map<String, String>;
          common.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"android": final YamlMap android}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["android"]! as Map<String, String>;
          android.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"ios": final YamlMap ios}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["ios"]! as Map<String, String>;
          ios.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"linux": final YamlMap linux}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["linux"]! as Map<String, String>;
          linux.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"windows": final YamlMap windows}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["windows"]! as Map<String, String>;
          windows.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"macos": final YamlMap macos}) {
          final tmp = (defs["defines"] as Map<String, Object>?)!["macos"]! as Map<String, String>;
          macos.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
        if (defines case {"web": final YamlMap web}) {
          final Map<String, String> tmp =
              (defs["defines"] as Map<String, Object>?)!["web"]! as Map<String, String>;
          web.forEach((k, v) => tmp.addAll({k.toString(): v.toString()}));
        }
      }
      if (yamlMap['hooks']['user_defines']['mnn'] case {"options": final YamlMap options}) {
        final Map<String, Object> tmp = (defs["options"] as Map<String, Object>?)!;
        options.forEach((k, v) => tmp.addAll({k.toString(): v as Object}));
      }
    }

    return defs;
  } catch (e) {
    // Return empty list in case of any error
    print('Error parsing exclude_modules: $e');
    return {};
  }
}
