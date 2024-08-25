function Model(Module)
{
    var model_new = Module.cwrap('model_new', 'number', []);
    var model_delete = Module.cwrap('model_delete', null, ['number']);
    var model_read_string = Module.cwrap('model_read_string', null, ['number', 'string']);
    var model_get_weights_names = Module.cwrap('model_get_weights_names', 'number', ['number']);
    var model_add_weights_file = Module.cwrap('model_add_weights_file', 'number', ['number', 'string', 'number']);

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
}
