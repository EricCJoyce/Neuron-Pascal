unit UpresLayerUnit;

{Neural Network library, by Eric C. Joyce

 An upres layer serves to prepare input for (transposed) convolution.
  s = stride
  p = padding

    input mat(X)         output for s = 1, p = 0        output for s = 1, p = 1
 [ x11 x12 x13 x14 ]    [ x11 0 x12 0 x13 0 x14 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x21 x22 x23 x24 ]    [  0  0  0  0  0  0  0  ]    [ 0 x11 0 x12 0 x13 0 x14 0 ]
 [ x31 x32 x33 x34 ]    [ x21 0 x22 0 x23 0 x24 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x41 x42 x43 x44 ]    [  0  0  0  0  0  0  0  ]    [ 0 x21 0 x22 0 x23 0 x24 0 ]
 [ x51 x52 x53 x54 ]    [ x31 0 x32 0 x33 0 x34 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x31 0 x32 0 x33 0 x34 0 ]
                        [ x41 0 x42 0 x43 0 x44 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x41 0 x42 0 x43 0 x44 0 ]
                        [ x51 0 x52 0 x53 0 x54 ]    [ 0  0  0  0  0  0  0  0  0 ]
                                                     [ 0 x51 0 x52 0 x53 0 x54 0 ]
                                                     [ 0  0  0  0  0  0  0  0  0 ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.}

interface

{**************************************************************************************************
 Constants  }

const
    FILL_ZERO   = 0;                                                { Fill strides or pad using zeroes }
    FILL_SAME   = 1;                                                { Fill strides or pad using duplicates of the nearest value }
    FILL_INTERP = 2;                                                { Fill strides or pad using bilinear interpolation }

    LAYER_NAME_LEN = 32;                                            { Length of a Layer 'name' string }

//{$DEFINE __UPRES_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    UpresParam = record
        stride_h: cardinal;                                         { Horizontal Stride: number of columns to put between input columns }
        stride_v: cardinal;                                         { Vertical Stride: number of rows to put between input rows }

        padding_h: cardinal;                                        { Horizontal Padding: depth of pixels appended to the source border, left and right }
        padding_v: cardinal;                                        { Vertical Padding: depth of pixels appended to the source border, top and bottom }

        sMethod: byte;                                              { In [FILL_ZERO, FILL_SAME, FILL_INTERP] }
        pMethod: byte;
    end;
    UpresLayer = record
        inputW: cardinal;                                           { Dimensions of the input }
        inputH: cardinal;
        params: array of UpresParam;                                { Array of Up-resolution parameters structures }
        nodes: cardinal;                                            { Number of up-resolution parameters in this layer }
        layerName: array [0..LAYER_NAME_LEN - 1] of char;
        outLen: cardinal;                                           { Length of the output buffer }
        out: array of double;
    end;

{**************************************************************************************************
 Prototypes  }

function add_UpResParams(const stride, padding: cardinal; var layer: UpresLayer): cardinal;
procedure setHorzStride_Upres(const stride, i: cardinal; var layer: UpresLayer);
procedure setVertStride_Upres(const stride, i: cardinal; var layer: UpresLayer);
procedure setHorzPad_Upres(const padding, i: cardinal; var layer: UpresLayer);
procedure setVertPad_Upres(const padding, i: cardinal; var layer: UpresLayer);
procedure setStrideMethod_Upres(const method: byte; const i: cardinal; var layer: UpresLayer);
procedure setPaddingMethod_Upres(const method: byte; const i: cardinal; var layer: UpresLayer);
procedure setName_Upres(const n: array of char; var layer: Pool2DLayer);
procedure print_Upres(const layer: UpresLayer);
function outputLen_Upres(const layer: UpresLayer): cardinal;
function run_Upres(const x: array of double; var layer: UpresLayer): cardinal;

implementation

{**************************************************************************************************
 Upres-Layers  }

{ Add an "up-ressing" to an existing UpresLayer.
  The new "up-ressing" shall have stride 'stride' and padding 'padding'. }
function add_UpResParams(const stride, padding: cardinal; var layer: UpresLayer): cardinal;
begin
    {$ifdef __UPRES_DEBUG}
    writeln('add_UpResParams(', stride, ', ', padding, ')');
    {$endif}

    layer.nodes += 1;                                               //  Increment the number of params/units
    SetLength(layer.params, layer.nodes);                           //  Allocate UpresParams in 'params' array

    layer.params[layer.nodes - 1].stride_h := stride;               //  Set newest up-ressing's stride (the same)
    layer.params[layer.nodes - 1].stride_v := stride;

    layer.params[layer.nodes - 1].padding_h := padding;             //  Set newest up-ressing's padding (the same)
    layer.params[layer.nodes - 1].padding_v := padding;

    layer.params[layer.nodes - 1].sMethod := FILL_ZERO;             //  Default to filling the strided rows and columns with zero
    layer.params[layer.nodes - 1].pMethod := FILL_ZERO;             //  Default to padding the input rows and columns with zero

    layer.outLen := outputLen_Upres(layer);
    SetLength(layer.out, layer.outLen);

    result := layer.nodes;
end;

procedure setHorzStride_Upres(const stride, i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
    begin
        layer.params[i].stride_h := stride;

        layer.outLen := outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

procedure setVertStride_Upres(const stride, i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
    begin
        layer.params[i].stride_v := stride;

        layer.outLen := outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

procedure setHorzPad_Upres(const padding, i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
    begin
        layer.params[i].padding_h := padding;

        layer.outLen := outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

procedure setVertPad_Upres(const padding, i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
    begin
        layer.params[i].padding_v := padding;

        layer.outLen := outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

procedure setStrideMethod_Upres(const method: byte; const i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
        layer.params[i].sMethod := method;
end;

procedure setPaddingMethod_Upres(const method: byte; const i: cardinal; var layer: UpresLayer);
begin
    if i < layer.nodes then
        layer.params[i].pMethod := method;
end;

procedure setName_Upres(const n: array of char; var layer: Pool2DLayer);
var
    i: byte;
    lim: byte;
begin
    if length(n) < LAYER_NAME_LEN then
        lim := length(n)
    else
        lim := LAYER_NAME_LEN;

    for i := 0 to lim - 1 do
        layer.layerName[i] := n[i];
    layer.layerName[lim] := chr(0);
end;

procedure print_Upres(const layer: UpresLayer);
var
    i: cardinal;
begin
    {$ifdef __UPRES_DEBUG}
    writeln('print_Upres()');
    {$endif}

    writeln('Input Shape = (', layer.inputW, ', ', layer.inputH, ')');

    for i := 0 to layer.nodes - 1 do                                //  Draw each up-ressing
    begin
        writeln('Parameters ', i);
        writeln('  H.stride  = ', layer.params[i].stride_h);
        writeln('  V.stride  = ', layer.params[i].stride_v);
        writeln('  H.padding = ', layer.params[i].padding_h);
        writeln('  V.padding = ', layer.params[i].padding_v);
        write('  Stride    = ');
        case layer.params[i].sMethod of
            FILL_ZERO:    writeln('zero');
            FILL_SAME:    writeln('same');
            FILL_INTERP:  writeln('interpolate');
        end;
        write('  Padding   = ');
        case layer.params[i].pMethod of
            FILL_ZERO:    writeln('zero');
            FILL_SAME:    writeln('same');
            FILL_INTERP:  writeln('interpolate');
        end;
    end;
end;

{ Return the layer's output length }
function outputLen_Upres(const layer: UpresLayer): cardinal;
var
    i: cardinal;
begin
    result := 0;

    for i := 0 to layer.nodes - 1 do
        result += (layer.inputW * (layer.params[i].stride_h + 1) - layer.params[i].stride_h + layer.params[i].padding_h + layer.params[i].padding_h) *
                  (layer.inputH * (layer.params[i].stride_v + 1) - layer.params[i].stride_v + layer.params[i].padding_v + layer.params[i].padding_v);
end;

{ Run the given input vector 'x' of length 'layer'.'inputW' * 'layer'.'inputH' through the UpresLayer 'layer'.
  The understanding for this function is that there is only one "color-channel."
  Output is stored internally in layer.out. }
function run_Upres(const x: array of double; var layer: UpresLayer): cardinal;
var
    i: cardinal;                                                    //  Up-ressing iterator
    o: cardinal;                                                    //  Output iterator
    x, y: cardinal;                                                 //  Iterators
    x_src, y_src: cardinal;                                         //  Used in zero-fill stride to iterate over source
    cache_w, cache_h: cardinal;                                     //  Dimensions of the inner-rectangle (without padding applied yet)
    output_w, output_h: cardinal;                                   //  Dimensions per up-ressing of the padded output
    x_prime, y_prime: double;                                       //  Inter-pixel locations in source
    a, b: double;                                                   //  Fractional parts of clipped doubles
    sc_inv_h, sc_inv_v: double;                                     //  Scaling factors of the inverse transformation
    val: double;                                                    //  Stores and compares neighboring pixel influence
    cache: array of double;                                         //  The "inner rectangle" we compute first
    ctr: cardinal;
begin
    {$ifdef __UPRES_DEBUG}
    writeln('run_Upres(', layer.inputW, ', ', layer.inputH, ')');
    {$endif}

    o := 0;

    for ctr := 0 to layer.outlen - 1 do                             //  Blank out the output buffer
        layer.out[ctr] := 0.0;

    for i := 0 to layer.n - 1 do                                    //  For each up-ressing, write the inner rectangle to cache and then wreath with padding.
    begin
                                                                    //  Compute the shape of the inner rectangle
        cache_w := layer.inputW * (layer.params[i].stride_h + 1) - layer.params[i].stride_h;
        cache_h := layer.inputH * (layer.params[i].stride_v + 1) - layer.params[i].stride_v;

        output_w := cache_w + 2 * layer.params[i].padding_h;        //  Compute the shape of the padded rectangle
        output_h := cache_h + 2 * layer.params[i].padding_v;
        SetLength(cache, cache_w * cache_h);                        //  Allocate cache for the inner rectangle

        ctr := 0;                                                   //  Reset counter: this now acts as our temporary output iterator

        if layer.params[i].sMethod = FILL_INTERP then               //  Fill strides using bilinear interpolation
        begin
            sc_inv_h := layer.inputW / cache_w;
            sc_inv_v := layer.inputH / cache_h;

            for y := 0 to cache_h - 1 do
            begin
                for x := 0 to cache_w - 1 do
                begin
                    x_prime := x * sc_inv_h;                        //  Where in the source does this pixel fall?
                    y_prime := y * sc_inv_v;

                    a := x_prime - Round(x_prime);                  //  Clip the fractional parts, store them in a and b:
                    b := y_prime - Round(y_prime);                  //  weigh the influences of neighboring pixels.

                    cache[ctr] := ((1.0 - a) * (1.0 - b)) * xvec[ y_prime      * layer.inputW + x_prime    ] +
                                  ((1.0 - a) * b)         * xvec[(y_prime + 1) * layer.inputW + x_prime    ] +
                                  (a * (1.0 - b))         * xvec[ y_prime      * layer.inputW + x_prime + 1] +
                                  (a * b)                 * xvec[(y_prime + 1) * layer.inputW + x_prime + 1];

                    ctr += 1;
                end;
            end;
        end
        else if layer.params[i].sMethod = FILL_SAME then            //  Fill strides in by duplicating the nearest source element
        begin
            sc_inv_h := layer.inputW / cache_w;
            sc_inv_v := layer.inputH / cache_h;

            for y := 0 to cache_h - 1 do
            begin
                for x := 0 to cache_w - 1 do
                begin
                    x_prime := x * sc_inv_h;                        //  Where in the source does this pixel fall?
                    y_prime := y * sc_inv_v;

                    a := x_prime - Round(x_prime);                  //  Clip the fractional parts, store them in a and b:
                    b := y_prime - Round(y_prime);                  //  weigh the influences of neighboring pixels.

                    val := ((1.0 - a) * (1.0 - b));                 //  Initial assumption: this pixel is nearest
                    cache[ctr]     := xvec[ y_prime      * layer.inputW + x_prime    ];

                    if ((1.0 - a) * b) > val then                   //  Does this pixel have greater influence?
                    begin
                        val := (1.0 - a) * b;
                        cache[ctr] := xvec[(y_prime + 1) * layer.inputW + x_prime    ];
                    end;
                    if (a * (1.0 - b)) > val then                   //  Does this pixel have greater influence?
                    begin
                        val := a * (1.0 - b);
                        cache[ctr] := xvec[ y_prime      * layer.inputW + x_prime + 1];
                    end;
                    if (a * b) > val then                           //  Does this pixel have greater influence?
                                                                    //  (No point storing 'val' anymore.)
                        cache[ctr] := xvec[(y_prime + 1) * layer.inputW + x_prime + 1];

                    ctr += 1;
                end;
            end;
        end
        else                                                        //  Fill strides in with zeroes
        begin
            x_src := 0;                                             //  Initialize source-iterators
            y_src := 0;

            y := 0
            while y < cache_h do
            begin
                x := 0;
                while x < cache_w do
                begin
                                                                    //  Copy source pixel
                    cache[ctr] := xvec[y_src * layer.inputW + x_src];
                    x_src += 1;                                     //  Increment source x-iterator
                    ctr += layer.params[i].stride_h + 1;            //  Advance output-iterator by horizontal stride
                    x += layer.params[i].stride_h + 1;
                end;
                x_src := 0;                                         //  Reset source x-iterator
                y_src += 1;                                         //  Increment source y-iterator
                ctr += (layer.params[i].stride_v + 1) * cache_w;    //  Advance output-iterator by vertical stride
                y += layer.params[i].stride_v + 1;
            end;
        end;

        if layer->params[i].pMethod <> FILL_ZERO then               //  Duplicate extrema
        begin
                                                                    //  First fill in the sides
            for y := layer.params[i].stride_v to output_h - layer.params[i].stride_v do
            begin
                for x := 0 to layer.params[i].stride_h - 1 do       //  Duplicate left side
                    layer.out[o + output_w * y + x] := cache[(y - layer.params[i].stride_v) * cache_w];
                                                                    //  Duplicate right side
                for x := layer.params[i].stride_h + cache_w to output_w - 1 do
                    layer.out[o + output_w * y + layer.params[i].stride_h + cache_w + x] := cache[(y - layer.params[i].stride_v) * cache_w + cache_w - 1];
            end;
                                                                    //  Then fill the top and bottom
            for y := 0 to layer.params[i].stride_v - 1 do           //  Fill top by referring to the first side-padded row
            begin
                for x := 0 to output_w - 1 do
                    layer.out[o + y * output_w + x] := layer.out[o + layer.params[i].stride_v * output_w + x];
            end;
                                                                    //  Fill bottom by referring to the last side-padded row
            for y := layer.params[i].stride_v + cache_h + 1 to output_h - 1 do
            begin
                for x := 0 to output_w - 1 do
                    layer.out[o + y * output_w + x] := layer.out[o + (layer.params[i].stride_v + cache_h) * output_w + x];
            end;
        end;
                                                                    //  Now, whether we had fancy padding or not, set cache into output buffer
        x_src := 0;                                                 //  Reset; these now iterate over the cached inner rectangle
        y_src := 0;
        for y := 0 to output_h - 1 do                               //  For every row in the padded output for the current up-ressing
        begin                                                       //  if we have passed the topmost padding and not yet reached the bottommost
            if (y >= layer.params[i].stride_v) and (y < output_h - layer.params[i].stride_v) then
            begin
                for x := 0 to output_w - 1 do                       //  For every column in the padded output for the current up-ressing
                begin                                               //  if we have passed the leftmost padding and not yet reached the rightmost
                    if (x >= layer.params[i].stride_h) and (x < output_w - layer.params[i].stride_h) then
                    begin
                                                                    //  Copy from cache
                        layer.out[o] := cache[y_src * cache_w + x_src];
                        x_src += 1;                                 //  Increment cache's x-iterator
                    end;
                    o += 1;                                         //  Increment output buffer iterator
                end;
                x_src := 0;                                         //  Reset cache's x-iterator
                y_src += 1;                                         //  Increment cache's y-iterator
            end
            else                                                    //  Otherwise, skip a whole output row
                o += output_w;
        end;

        SetLength(cache, 0);                                        //  Release
    end;

    result := layer.outLen;
end;
