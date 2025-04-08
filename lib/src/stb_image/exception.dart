class StbiException implements Exception {
  StbiException(this.message);
  final String message;
  @override
  String toString() => "StbiException($message)";
}
