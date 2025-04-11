class MNNException implements Exception {
  final String message;
  MNNException(this.message);
  @override
  String toString() {
    return '(MNNException: $message)';
  }
}
