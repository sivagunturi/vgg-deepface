class Format:
    InputShapeAlign = 25
    NameAlign = 20
    FiltersAlign = 10
    PaddingAlign = 20
    StridesAlign = 18
    KernelSizeAlign = 20
    OutputShapeAlign = 25
    ParamsAlign = 30
    LineLength = 200


class LayerName:
    LayerName = "name"
    Filters = "filters"
    Padding = "padding"
    Strides = "strides"
    KernelSize = "kernel_size"
    InputShape = "input_shape"
    OutputShape = "output_shape"
    Params = "params => (k*k*input_shape[3] + 1) * f"
    PipeSeperator = "|"


class Colors:

    green = '\x1b[92m {}\x1b[00m'
    blue = '\033[94m {}\033[00m'
    yellow = '\033[93m {}\033[00m'
    bold = '\033[1m {}\033[00m'


def model_summary(model):
    print(repr('=' * Format.LineLength).replace("'", ''))
    first_row = '|' + Colors.bold.format(
        LayerName.InputShape.center(
            Format.InputShapeAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.LayerName.center(
            Format.NameAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.Filters.center(
            Format.FiltersAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.Padding.center(
            Format.PaddingAlign + 10)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.Strides.center(
            Format.StridesAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.KernelSize.center(
            Format.KernelSizeAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.OutputShape.center(
            Format.OutputShapeAlign)) + LayerName.PipeSeperator + Colors.bold.format(
        LayerName.Params.center(
            Format.ParamsAlign))
    print(first_row)
    print(repr('=' * Format.LineLength).replace("'", ''))
    layer_count = 0
    total_params = 0
    for layer in model.layers:
        input_shape = ""
        input_shape += '[' + str(layer_count) + ']'
        input_shape += str(layer.input_shape)
        input_shape = input_shape.center(Format.InputShapeAlign)
        input_shape = Colors.green.format(input_shape)
        input_shape += LayerName.PipeSeperator

        name = ''
        if hasattr(layer, LayerName.LayerName):
            name += layer.name
            name = name.center(Format.NameAlign)
            name = Colors.blue.format(name)
        filters = ''
        if hasattr(layer, LayerName.Filters):
            filters += str(layer.filters)
        filters = filters.center(Format.FiltersAlign)
        filters = Colors.blue.format(filters)
        padding = ''
        if hasattr(layer, LayerName.Padding):
            padding += str(layer.padding)
        padding = padding.center(Format.ParamsAlign)
        padding = Colors.blue.format(padding)
        strides = ''
        if hasattr(layer, LayerName.Strides):
            strides += str(layer.strides)
        strides = strides.center(Format.StridesAlign)
        strides = Colors.blue.format(strides)
        kernel_size = ''
        if hasattr(layer, LayerName.KernelSize):
            kernel_size += str(layer.kernel_size)
        kernel_size = kernel_size.center(Format.KernelSizeAlign)
        kernel_size = Colors.blue.format(kernel_size)

        output_shape = str(layer.output_shape)
        output_shape = output_shape.center(Format.OutputShapeAlign)
        output_shape = Colors.green.format(output_shape)

        layer_params = 0
        if hasattr(layer, LayerName.Filters):
            k = layer.kernel_size[0]
            f = layer.filters
            layer_params = (k * k * layer.input_shape[3] + 1) * f
            params = str(layer_params)
        else:
            params = "0"
        total_params += layer_params
        params = params.center(Format.ParamsAlign)
        params = Colors.yellow.format(params)

        output_buffer = LayerName.PipeSeperator + input_shape + name + LayerName.PipeSeperator + filters + LayerName.PipeSeperator + padding + \
            LayerName.PipeSeperator + strides + LayerName.PipeSeperator + kernel_size + LayerName.PipeSeperator + output_shape + LayerName.PipeSeperator + params

        print(output_buffer)
        print(repr('_' * Format.LineLength).replace("'", ''))

        layer_count += 1

    total_params = "Total params = " + str(total_params)
    total_params = total_params.rjust(Format.LineLength - 10)
    total_params = Colors.bold.format(total_params)
    print(total_params)
