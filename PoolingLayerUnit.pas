unit PoolingLayerUnit;

{Neural Network library, by Eric C. Joyce

 Model a Pooling Layer as 2D input dimensions and an array of 2D pools.
  inputW = width of the input
  inputH = height of the input

 Each pool has a 2D shape, two dimensions for stride, and a function/type:
  stride_h = horizontal stride of the pool
  stride_v = vertical stride of the pool
  f = [MAX_POOL, AVG_POOL, MIN_POOL, MEDIAN_POOL]

    input mat(X)          pool     output for s = (1, 1)     output for s = (2, 2)
 [ x11 x12 x13 x14 ]    [ . . ]   [ y11  y12  y13 ]         [ y11  y12 ]
 [ x21 x22 x23 x24 ]    [ . . ]   [ y21  y22  y23 ]         [ y21  y22 ]
 [ x31 x32 x33 x34 ]              [ y31  y32  y33 ]
 [ x41 x42 x43 x44 ]              [ y41  y42  y43 ]
 [ x51 x52 x53 x54 ]

 Pools needn't be arranged from smallest to largest or in any order.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.}

interface

{**************************************************************************************************
 Constants  }

const
    MAX_POOL    = 0;
    MIN_POOL    = 1;
    AVG_POOL    = 2;
    MEDIAN_POOL = 3;

    LAYER_NAME_LEN = 32;                                            { Length of a Layer 'name' string }

//{$DEFINE __POOL_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    Pool2D = record
        w: cardinal;                                                { Width of the filter }
        h: cardinal;                                                { Height of the filter }
        stride_h: cardinal;                                         { Stride by which we move the filter left to right }
        stride_v: cardinal;                                         { Stride by which we move the filter top to bottom }
        f: byte;                                                    { Function flag, in [MAX_POOL, MIN_POOL, AVG_POOL, MEDIAN_POOL] }
    end;
    Pool2DLayer = record
        inputW: cardinal;                                           { Dimensions of the input }
        inputH: cardinal;
        nodes: cardinal;                                            { Number of processing units in this layer = number of filters in this layer }
        pools: array of Pool2D;                                     { Array of 2D pools }
        layerName: array [0..LAYER_NAME_LEN - 1] of char;
        outLen: cardinal;                                           { Length of the output buffer }
        out: array of double;
    end;

{**************************************************************************************************
 Prototypes  }

function add_2DPool(const poolw, poolh: cardinal; var layer: Pool2DLayer): cardinal;
procedure setW_Pool2D(const width, i: cardinal; var layer: Pool2DLayer);
procedure setH_Pool2D(const height, i: cardinal; var layer: Pool2DLayer);
procedure setHorzStride_Pool2D(const stride, i; var layer: Pool2DLayer);
procedure setVertStride_Pool2D(const stride, i; var layer: Pool2DLayer);
procedure setFunc_Pool2D(const func: byte; const i: cardinal; var layer: Pool2DLayer);
procedure setName_Pool2D(const n: array of char; var layer: Pool2DLayer);
procedure print_Pool2D(const layer: Pool2DLayer);
function outputLen_Pool2D(const layer: Pool2DLayer): cardinal;
function run_Pool2D(const xvec: array of double; var layer: Pool2DLayer): cardinal;
procedure pooling_quicksort(const descending: boolean; var arr: array of double; const lo, hi: cardinal);
function pooling_partition(const descending: boolean; var arr: array of double; const lo, hi: cardinal): cardinal;

implementation

{**************************************************************************************************
 Pooling-Layers  }

{ Add a Pool2D to an existing Pool2DLayer.
  The new pool shall have dimensions 'poolw' by 'poolh'. }
function add_2DPool(const poolw, poolh: cardinal; var layer: Pool2DLayer): cardinal;
begin
    {$ifdef __POOL_DEBUG}
    writeln('add_2DPool(', poolw, ', ', poolh, ')');
    {$endif}

    layer.nodes += 1;                                               //  Increment the number of pools/units
    SetLength(layer.pools, layer.nodes);                            //  Allocate pool in 'pools' array

    layer.pools[layer.nodes - 1].w := poolw;                        //  Set newest filter's dimensions
    layer.pools[layer.nodes - 1].h := poolh;
    layer.pools[layer.nodes - 1].stride_h := 1;                     //  Default to stride (1, 1)
    layer.pools[layer.nodes - 1].stride_v := 1;
    layer.pools[layer.nodes - 1].f := MAX_POOL;                     //  Default to max-pooling

    layer.outLen := outputLen_Pool2D(layer);
    SetLength(layer.out, layer.outLen);

    result := layer.nodes;
end;

{ (Re)Set the width of the i-th pool in the given layer. }
procedure setW_Pool2D(const width, i: cardinal; var layer: Pool2DLayer);
begin
    if i < layer.nodes then
    begin
        layer.pools[i].w := width;

        layer.outLen := outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

{ (Re)Set the height of the i-th pool in the given layer. }
procedure setH_Pool2D(const height, i: cardinal; var layer: Pool2DLayer);
begin
    if i < layer.nodes then
    begin
        layer.pools[i].h := height;

        layer.outLen := outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

{ (Re)Set the horizontal stride of the i-th pool in the given layer. }
procedure setHorzStride_Pool2D(const stride, i: cardinal; var layer: Pool2DLayer);
begin
    if i < layer.nodes then
    begin
        layer.pools[i].stride_h := stride;

        layer.outLen := outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

{ (Re)Set the vertical stride of the i-th pool in the given layer. }
procedure setVertStride_Pool2D(const stride, i: cardinal; var layer: Pool2DLayer);
begin
    if i < layer.nodes then
    begin
        layer.pools[i].stride_v := stride;

        layer.outLen := outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        SetLength(layer.out, layer.outLen);
    end;
end;

{ Set the function for the i-th pool in the given layer. }
procedure setFunc_Pool2D(const func: byte; const i: cardinal; var layer: Pool2DLayer);
begin
    if i < layer.nodes then
        layer.pools[i].f := func;
end;

{ Set the name of the given Pooling Layer }
procedure setName_Pool2D(const n: array of char; var layer: Pool2DLayer);
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

{ Print the details of the given Pool2DLayer 'layer' }
procedure print_Pool2D(const layer: Pool2DLayer);
var
    i, x, y: cardinal;
begin
    {$ifdef __POOL_DEBUG}
    writeln('print_Pool2D()');
    {$endif}

    writeln('Input Shape = (', layer.inputW, ', ', layer.inputH, ')');

    for i := 0 to layer.nodes - 1 do                                //  Draw each pool
    begin
        writeln('Pool ', i);
        for y := 0 to layer.pools[i].h - 1 do
        begin
            write('  [');
            for x := 0 to layer.pools[i].w - 1 do
                write(' . ');
            write(']');
            if y < layer.pools[i].h - 1 then
              writeln('');
        end;
        write('  Func:  ');
        case layer.pools[i].f of
            MAX_POOL:     write('max.  ');
            MIN_POOL:     write('min.  ');
            AVG_POOL:     write('avg.  ');
            MEDIAN_POOL:  write('med.  ');
        end;
        writeln('Stride: (', layer.pools[i].stride_h, ', ', layer.pools[i].stride_v,')');
    end;
end;

{ Return the layer's output length }
function outputLen_Pool2D(const layer: Pool2DLayer): cardinal;
var
    i: cardinal;
begin
    result := 0;

    for i := 0 to layer.nodes - 1 do
        result += (floor((layer.inputW - layer.pools[i].w + 1) / layer.pools[i].stride_h) *
                   floor((layer.inputH - layer.pools[i].h + 1) / layer.pools[i].stride_v));

    return ctr;
end;

{ Run the given input vector 'x' of length 'layer'.'inputW' * 'layer'.'inputH' through the Pool2DLayer 'layer'.
  The understanding for this function is that pooling never runs off the edge of the input, and that there is
  only one "color-channel."
  Output is stored internally in layer.out. }
function run_Pool2D(const xvec: array of double; var layer: Pool2DLayer): cardinal;
var
    i: cardinal;                                                    //  Pool array iterator
    o: cardinal;                                                    //  Output iterator
    ctr: cardinal;                                                  //  Only used in median pooling
    x, y: cardinal;                                                 //  2D input iterators
    m, n: cardinal;                                                 //  2D pool iterators

    cache: array of double;                                         //  Intermediate buffer
    cacheLen: cardinal;                                             //  Length of that buffer
    cacheLenEven: boolean;
    index: cardinal;                                                //  Used in median pooling
    val: double;
begin
    {$ifdef __POOL_DEBUG}
    writeln('run_Pool2D(', layer.inputW, ', ', layer.inputH, ')');
    {$endif}

    i := 0;
    o := 0;
    cacheLenEven := false;

    for i := 0 to layer.nodes - 1 do                                //  For each pool
    begin
        case layer.pools[i].f of                                    //  Prefer one "if" per layer to one "if" per iteration
            MAX_POOL:
            begin
                y := 0;
                while y <= layer->inputH - layer->pools[i].h do
                begin
                    x := 0;
                    while x <= layer.inputW - layer.pools[i].w do
                    begin
                        val := -Infinity;
                        for n := 0 to layer.pools[i].h - 1 do
                        begin
                            for m := 0 to layer.pools[i].w - 1 do
                            begin
                                if xvec[(y + n) * layer.inputW + x + m] > val then
                                    val := xvec[(y + n) * layer.inputW + x + m];
                            end;
                        end;
                        layer.out[o] := val;
                        o += 1;
                        x += layer.pools[i].stride_h;
                    end;
                    y += layer.pools[i].stride_v;
                end;
            end;
            MIN_POOL:
            begin
                y := 0;
                while y <= layer->inputH - layer->pools[i].h do
                begin
                    x := 0;
                    while x <= layer.inputW - layer.pools[i].w do
                    begin
                        val := Infinity;
                        for n := 0 to layer.pools[i].h - 1 do
                        begin
                            for m := 0 to layer.pools[i].w - 1 do
                            begin
                                if xvec[(y + n) * layer.inputW + x + m] < val then
                                    val := xvec[(y + n) * layer.inputW + x + m];
                            end;
                        end;
                        layer.out[o] := val;
                        o += 1;
                        x += layer.pools[i].stride_h;
                    end;
                    y += layer.pools[i].stride_v;
                end;
            end;
            AVG_POOL:
            begin
                y := 0;
                while y <= layer->inputH - layer->pools[i].h do
                begin
                    x := 0;
                    while x <= layer.inputW - layer.pools[i].w do
                    begin
                        val := 0.0;
                        for n := 0 to layer.pools[i].h - 1 do
                        begin
                            for m := 0 to layer.pools[i].w - 1 do
                            begin
                                if xvec[(y + n) * layer.inputW + x + m] < val then
                                    val += xvec[(y + n) * layer.inputW + x + m];
                            end;
                        end;
                        layer.out[o] := val / (layer.pools[i].w * layer.pools[i].h);
                        o += 1;
                        x += layer.pools[i].stride_h;
                    end;
                    y += layer.pools[i].stride_v;
                end;
            end;
            MEDIAN_POOL:
            begin
                cacheLen := layer.pools[i].w * layer.pools[i].h;
                SetLength(cache, cacheLen);
                if (cacheLen mod 2) = 0 then
                    cacheLenEven = true
                else
                    cacheLenEven = false;
                if cacheLenEven then
                    index := cacheLen / 2 - 1
                else
                    index := (cacheLen - 1) / 2;

                y := 0;
                while y <= layer.inputH - layer.pools[i].h do
                begin
                    x := 0;
                    while x <= layer.inputW - layer.pools[i].w do
                    begin
                        ctr := 0;
                        for n := 0 to layer.pools[i].h - 1 do
                        begin
                            for m := 0 to layer.pools[i].w - 1 do
                            begin
                                cache[ctr] := xvec[(y + n) * layer.inputW + x + m];
                                ctr += 1;
                            end;
                        end;

                        pooling_quicksort(true, cache, 0, cacheLen - 1);

                        if cacheLenEven then
                            layer.out[o] := 0.5 * (cache[index] + cache[index + 1])
                        else
                            layer.out[o] = cache[index];

                        o += 1;
                        x += layer.pools[i].stride_h;
                    end;
                    y += layer.pools[i].stride_v;
                end;
                SetLength(cache, 0);
            end;
        end;
    end;

    result := layer.outLen;
end;

procedure pooling_quicksort(const descending: boolean; var arr: array of double; const lo, hi: cardinal);
var
    p: cardinal;
begin
    if lo < hi then
    begin
        p := pooling_partition(descending, arr, lo, hi);

        if p > 0 then                                               //  PREVENT ROLL-OVER TO UINT_MAX
          pooling_quicksort(descending, arr, lo, p - 1);            //  Left side: start quicksort
        if p < maxint then                                          //  PREVENT ROLL-OVER TO 0x0000
          pooling_quicksort(descending, arr, p + 1, hi);            //  Right side: start quicksort
    end;
end;

function pooling_partition(const descending: boolean; var arr: array of double; const lo, hi: cardinal): cardinal;
var
    pivot: double;
    i, j: cardinal;
    tmpFloat: cardinal;
    trigger: boolean;
begin
    pivot := arr[hi];
    i := lo;

    for j := lo to hi - 1 do
    begin
        if descending then                                          //  Sort descending
        begin
            if arr[j] > pivot then
                trigger := true
            else
                trigger := false;
        end
        else                                                        //  Sort ascending
        begin
            if arr[j] < pivot then
                trigger := true
            else
                trigger := false;
        end;

        if trigger then
        begin
            tmpFloat := arr[i];                                     //  Swap a[i] with a[j]
            arr[i]   := arr[j];
            arr[j]   := tmpFloat;

            i += 1;
        end;
    end;

    tmpFloat := arr[i];                                             //  Swap a[i] with a[hi]
    arr[i]   := arr[hi];
    arr[hi]  := tmpFloat;

    result := i;
end;
