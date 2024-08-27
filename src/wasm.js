function Model(Module)
{
    var model_new = Module.cwrap('model_new', 'number', []);
    var model_delete = Module.cwrap('model_delete', null, ['number']);
    var model_read_string = Module.cwrap('model_read_string', null, ['number', 'string']);
    var model_get_weights_names = Module.cwrap('model_get_weights_names', 'number', ['number']);
    var model_add_weights_file = Module.cwrap('model_add_weights_file', 'number', ['number', 'string', 'number']);
    var model_add_tensor = Module.cwrap('model_add_tensor', 'number', ['number', 'string', 'number', 'number']);
    var model_get_tensor = Module.cwrap('model_get_tensor', 'number', ['number', 'string']);
    var model_run = Module.cwrap('model_run', null, ['number']);
    var model_clear_tensors = Module.cwrap('model_clear_tensors', null, ['number']);
    var model_set_option = Module.cwrap('model_set_option', null, ['number', 'string', 'number']);

    var instance = model_new();

    this.delete = function () {
        model_delete(instance);
    };

    this.read_string = function (str) {
        model_read_string(instance, str);
    };

    this.get_weights_names = function () {
        var buf = model_get_weights_names(instance);
        var ret = Module.UTF8ToString(buf);
        Module._free(buf);
        return ret;
    };

    this.add_weights_file = function (name, buffer) {
        var ta = new Float32Array(buffer);
        var ptr = model_add_weights_file(instance, name, ta.length);
        Module.HEAPU8.set(new Uint8Array(ta.buffer), ptr >>> 0);
    };

    this.add_tensor = function (name, shape, buffer) {
        var ta32 = new Uint32Array(shape);
        var ta8 = new Uint8Array(ta32.buffer);
        var buf = Module._malloc(ta8.length);
        Module.HEAPU8.set(ta8, buf >>> 0);
        var ptr = model_add_tensor(instance, name, ta32.length, buf);
        Module._free(buf);
        var taData = new Float32Array(buffer);
        Module.HEAPU8.set(new Uint8Array(taData.buffer), ptr >>> 0);
    };

    this.get_tensor = function (name) {
        var buf = model_get_tensor(instance, name);
        if (!buf)
            return null;
        var ta = new Uint32Array(Module.HEAPU8.buffer, buf >>> 0, 4);
        var dims_num = ta[0];
        var dims = ta[1];
        var data_num = ta[2];
        var data = ta[3];
        Module._free(buf);
        return {
            shape: new Uint32Array(Module.HEAPU8.buffer, dims, dims_num),
            data: new Float32Array(Module.HEAPU8.buffer, data, data_num)
        };
    };

    this.run = function () {
        model_run(instance);
    };

    this.clear_tensors = function () {
        model_clear_tensors(instance);
    };

    this.set_option = function (name, value) {
        model_set_option(instance, name, value ? 1 : 0);
    };
}
