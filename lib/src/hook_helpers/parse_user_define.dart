import 'dart:io';
import 'package:yaml/yaml.dart';

/// Parse the user defined exclude modules from pubspec.yaml
///
/// Returns a list of excluded module names
Map<String, String> parseUserDefinedOptions(String pubspecPath) {
  try {
    // Read the pubspec.yaml file
    final File file = File(pubspecPath);
    if (!file.existsSync()) {
      return {};
    }

    // Parse the YAML content
    final String yamlContent = file.readAsStringSync();
    final dynamic yamlMap = loadYaml(yamlContent);

    // Navigate to the hooks.user_defines.mnn.exclude_modules section
    if (yamlMap is YamlMap &&
        yamlMap['hooks'] is YamlMap &&
        yamlMap['hooks']['user_defines'] is YamlMap &&
        yamlMap['hooks']['user_defines']['mnn'] is YamlMap &&
        yamlMap['hooks']['user_defines']['mnn']['defines'] is YamlMap) {
      final YamlMap excludeModules = yamlMap['hooks']['user_defines']['mnn']['defines'] as YamlMap;

      return excludeModules.map((k, v) => MapEntry(k.toString(), v.toString()));
    }

    return {};
  } catch (e) {
    // Return empty list in case of any error
    print('Error parsing exclude_modules: $e');
    return {};
  }
}
