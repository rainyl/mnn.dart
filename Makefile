ffigen:
	dart run ffigen --config ffigen.yaml

coverage:
	flutter test --coverage

test:
	flutter test

copy-debug:
	cp -f src/cmake-build-debug/libmnn_c_api.dylib .dart_tool/lib/libmnn_c_api.dylib
