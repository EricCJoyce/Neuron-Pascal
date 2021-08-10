unit DenseLayerUnit;

{Neural Network library, by Eric C. Joyce

 Model a Dense Layer as two matrices and two vectors:

    input vec(x)         weights W             masks M
 [ x1 x2 x3 x4 1 ]  [ w11 w12 w13 w14 ]  [ m11 m12 m13 m14 ]
                    [ w21 w22 w23 w24 ]  [ m21 m22 m23 m24 ]
                    [ w31 w32 w33 w34 ]  [ m31 m32 m33 m34 ]
                    [ w41 w42 w43 w44 ]  [ m41 m42 m43 m44 ]
                    [ w51 w52 w53 w54 ]  [  1   1   1   1  ]

                    activation function
                         vector f
               [ func1 func2 func3 func4 ]

                     auxiliary vector
                          alpha
               [ param1 param2 param3 param4 ]

 Broadcast W and M = W'
 vec(x) dot W' = x'
 vec(output) is func[i](x'[i], param[i]) for each i

 Not all activation functions need a parameter. It's just a nice feature we like to offer.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.}

interface

{**************************************************************************************************
 Constants  }

const
    RELU                = 0;                                        { [ 0.0, inf) }
    LEAKY_RELU          = 1;                                        { (-inf, inf) }
    SIGMOID             = 2;                                        { ( 0.0, 1.0) }
    HYPERBOLIC_TANGENT  = 3;                                        { [-1.0, 1.0] }
    SOFTMAX             = 4;                                        { [ 0.0, 1.0] }
    SYMMETRICAL_SIGMOID = 5;                                        { (-1.0, 1.0) }
    THRESHOLD           = 6;                                        {   0.0, 1.0  }
    LINEAR              = 7;                                        { (-inf, inf) }

    LAYER_NAME_LEN      = 32;                                       { Length of a Layer 'name' string }

//{$DEFINE __DENSE_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    DenseLayer = record
        inputs: cardinal;                                           { Number of inputs--NOT COUNTING the added bias-1 }
        nodes: cardinal;                                            { Number of processing units in this layer }
        W: array of double;                                         { ((inputs + 1) x nodes) matrix }
        M: array of real;                                           { ((inputs + 1) x nodes) matrix, all either 0.0 or 1.0 }
        functions: array of byte;                                   { n-array }
        alphas: array of double;                                    { n-array }
        layerName: array [0..LAYER_NAME_LEN - 1] of char;
        out: array of double;
    end;

{**************************************************************************************************
 Prototypes  }

                                                                    { Set entirety of layer's weight matrix }
procedure setW_Dense(const w: array of double; var layer: DenseLayer);
                                                                    { Set entirety of weights for i-th column/neuron/unit }
procedure setW_i_Dense(const w: array of double; const i: cardinal; var layer: DenseLayer);
                                                                    { Set element [i, j] of layer's weight matrix }
procedure setW_ij_Dense(const w: double; const i: cardinal; const j: cardinal; var layer: DenseLayer);
                                                                    { Set entirety of layer's mask matrix }
procedure setM_Dense(const m: array of boolean; var layer: DenseLayer);
                                                                    { Set entirety of masks for i-th column/neuron/unit }
procedure setM_i_Dense(const m: array of boolean; const i: cardinal; var layer: DenseLayer);
                                                                    { Set element [i, j] of layer's mask matrix }
procedure setM_ij_Dense(const m: boolean, const i, j: cardinal; var layer: DenseLayer);
                                                                    { Set activation function of i-th neuron/unit }
procedure setF_i_Dense(const func: byte, const i: cardinal; var layer: DenseLayer);
                                                                    { Set activation function auxiliary parameter of i-th neuron/unit }
procedure setA_i_Dense(const a: double; const i: cardinal; var layer: DenseLayer);
procedure setName_Dense(const n: array of char; var layer: DenseLayer);
procedure print_Dense(const layer: DenseLayer);
function outputLen_Dense(const layer: DenseLayer): cardinal;
function run_Dense(const x: array of double; var layer: DenseLayer): cardinal;

implementation

{**************************************************************************************************
 Dense-Layers  }

{ Set entirety of layer's weight matrix.
  Input buffer 'w' is expected to be ROW-MAJOR
       weights W
  [ w0  w1  w2  w3  ]
  [ w4  w5  w6  w7  ]
  [ w8  w9  w10 w11 ]
  [ w12 w13 w14 w15 ]
  [ w16 w17 w18 w19 ]  <--- biases }
procedure setW_Dense(const w: array of double; var layer: DenseLayer);
var
    i: cardinal;
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setW_Dense()');
    {$endif}

    for i := 0 to (layer.inputs + 1) * layer.nodes - 1 do
        layer.W[i] := w[i];
end;

{ Set entirety of weights for i-th column/neuron/unit. }
procedure setW_i_Dense(const w: array of double; const i: cardinal; var layer: DenseLayer);
var
    j: cardinal;
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setW_i_Dense(', i, ')');
    {$endif}

    for i := 0 to layer.inputs - 1 do
        layer.W[j * layer.nodes + i] := w[j];
end;

{ Set unit[i], weight[j] of the given layer }
procedure setW_ij_Dense(const w: double; const i: cardinal; const j: cardinal; var layer: DenseLayer);
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setW_ij_Dense(', i, ', ', j, ')');
    {$endif}

    if j * layer.nodes + i < (layer.inputs + 1) * layer.nodes then
        layer.W[j * layer.nodes + i] := w;
end;

{ Set entirety of layer's mask matrix
  Input buffer 'm' is expected to be ROW-MAJOR
       masks M
  [ m0  m1  m2  m3  ]
  [ m4  m5  m6  m7  ]
  [ m8  m9  m10 m11 ]
  [ m12 m13 m14 m15 ]
  [ m16 m17 m18 m19 ]  <--- biases }
procedure setM_Dense(const m: array of boolean; var layer: DenseLayer);
var
    i: cardinal;
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setM_Dense()');
    {$endif}

    for i := 0 to (layer.inputs + 1) * layer.nodes - 1 do
    begin
        if m[i] then
            layer.M[i] := 1.0
        else
            layer.M[i] := 0.0;
    end;
end;

{ Set entirety of masks for i-th column/neuron/unit }
procedure setM_i_Dense(const m: array of boolean; const i: cardinal; var layer: DenseLayer);
var
    j: cardinal;
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setM_i_Dense(', i, ')');
    {$endif}

    for j := 0 to layer.inputs do
    begin
        if m[j] then
            layer.M[j * layer.nodes + i] := 1.0
        else
            layer.M[j * layer.nodes + i] := 0.0;
    end;
end;

{ Set unit[i], weight[j] of the given layer }
procedure setM_ij_Dense(const m: boolean; const i, j: cardinal; var layer: DenseLayer);
begin
    {$ifdef __DENSE_DEBUG}
    if m then
        writeln('setM_ij_Dense(LIVE, ', i, ', ', j, ')')
    else
        writeln('setM_ij_Dense(MASKED, ', i, ', ', j, ')');
    {$endif}

    if j * layer.nodes + i < (layer.inputs + 1) * layer.nodes) then
    begin
        if m then
            layer.M[j * layer.nodes + i] := 1.0
        else
            layer.M[j * layer.nodes + i] := 0.0;
    end;
end;

{ Set the activation function for unit[i] of the given layer }
procedure setF_i_Dense(const func: byte; const i: cardinal; var layer: DenseLayer);
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setF_i_Dense(', func, ', ', i, ')');
    {$endif}

    if i < layer.nodes then
        layer.functions[i] := func;
end;

{ Set the activation function parameter for unit[i] of the given layer }
procedure setA_i_Dense(const a: double; const i: cardinal; var layer: DenseLayer);
begin
    {$ifdef __DENSE_DEBUG}
    writeln('setA_i_Dense(', a, ', ', i, ')');
    {$endif}

    if i < layer.nodes then
        layer.alphas[i] := a;
end;

{ Set the name of the given Dense Layer }
procedure setName_Dense(const n: array of char; var layer: DenseLayer);
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

{ Print the details of the given DenseLayer 'layer' }
procedure print_Dense(const layer: DenseLayer);
var
    i, j: cardinal;
begin
    unsigned int i, j;

    {$ifdef __DENSE_DEBUG}
    writeln('print_Dense()');
    {$endif}

    for i := 0 to layer.inputs do
    begin
        if i = layer.inputs then
            write('bias [')
        else
            write('     [');

        for j := 0 to layer.nodes - 1 do
        begin
            if layer.W[i * layer.nodes + j] >= 0.0 then
                write(' ', layer.W[i * layer.nodes + j], ' ')
            else
                write(layer.W[i * layer.nodes + j], ' ');
        end;
        writeln(']');
    end;
    write('f = [');
    for i := 0 to layer.nodes - 1 do
    begin
        case (layer.functions[i]) of
            RELU:                write('ReLU   ');
            LEAKY_RELU:          write('L.ReLU ');
            SIGMOID:             write('Sig.   ');
            HYPERBOLIC_TANGENT:  write('tanH   ');
            SOFTMAX:             write('SoftMx ');
            SYMMETRICAL_SIGMOID: write('SymSig ');
            THRESHOLD:           write('Thresh ');
            LINEAR:              write('Linear ');
    end;
    writeln(']');
    write('a = [');
    for i := 0 to layer.nodes - 1 do
        write(layer.alphas[i], ' ');
    writeln(']');
end;

{ Return the layer's output length
  (For Dense layers, this is the number of units) }
function outputLen_Dense(const layer: DenseLayer): cardinal;
begin
    {$ifdef __DENSE_DEBUG}
    writeln('outputLen_Dense()');
    {$endif}

    result := layer.nodes;
end;

{ Run the given input vector 'x' of length 'layer'.'inputs' through the DenseLayer 'layer'.
  Output is stored internally in layer.out. }
function run_Dense(const x: array of double; var layer: DenseLayer): cardinal;
var
    xprime: array [0..layer.inputs] of double;                      //  Input vector augmented with additional (bias) 1.0
                                                                    //  (1 * (length-of-input + 1))
                                                                    //  ((length-of-input + 1) * nodes)
    Wprime: array [0..(layer.inputs + 1) * layer.nodes - 1] of double;
    softmaxdenom: double;                                           //  Accumulate exp()'s to normalize any softmax
    i: cardinal;
begin
end;
    {$ifdef __DENSE_DEBUG}
    writeln('run_Dense(', layer.inputs, ')');
    {$endif}

    softmaxdenom := 0.0;

    for i := 0 to layer.inputs - 1 do                               //  Append 1.0 to input vector
        xprime[i] := x[i];
    xprime[i] := 1.0;

    //                       weights W                                                  masks M
    //     i = 0 ----------------------------> layer->nodes        i = 0 ----------------------------> layer->nodes
    //   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]
    //   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]
    //   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]
    //   |   [ ...      ...                  ...             ]   |   [ ...      ...                  ...             ]
    //   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]
    // len+1 [ A+len+1  A+((len+1)*i)+len+1  A+((len+1)*i)+j ] len+1 [  1        1                    1              ]
    for i := 0 to (layer.inputs + 1) * layer.nodes - 1 do           //  Broadcast weights and masks into W'
        Wprime[i] := layer.W[i] * layer.[i];
                                                                    //  Dot-product xprime Wprime ---> layer->out
    cblas_dgemv(CblasRowMajor,                                      //  The order in which data in Wprime are stored
                CblasNoTrans,                                       //  Transpose
                layer.inputs + 1,                                   //  Number of ROWS in Wprime = number of inputs + 1 row of biases
                layer.nodes,                                        //  Number of COLUMNS in Wprime = number of layer units
                1.0,                                                //  alpha (ignore this)
                Wprime, layer.nodes,                                //  Stride in Wprime equals the number of COLUMNS when order = CblasRowMajor
                xprime, 1,                                          //  Stride in xprime
                0.0, layer.out, 1);

    for i := 0 to layer.nodes - 1 do                                //  In case one of the units is a softmax unit,
        softmaxdenom := softmaxdenom + exp(layer.out[i]);           //  compute all exp()'s so we can sum them.

    for i := 0 to layer.nodes - 1 do                                //  Run each element in out through appropriate function
    begin                                                           //  with corresponding parameter
        case (layer.functions[i]) of
            RELU:                 if layer.out[i] < 0.0 then layer.out[i] := 0.0;
            LEAKY_RELU:           if layer.out[i] < 0.0 then layer.out[i] := layer.out[i] * layer.alphas[i];
            SIGMOID:              layer.out[i] := 1.0 / (1.0 + exp(-layer.out[i] * layer.alphas[i]));
            HYPERBOLIC_TANGENT:   layer.out[i] := (2.0 / (1.0 + exp(-2.0 * layer.out[i] * layer.alphas[i]))) - 1.0;
            SOFTMAX:              layer.out[i] := exp(layer.out[i]) / softmaxdenom;
            SYMMETRICAL_SIGMOID:  layer.out[i] := (1.0 - exp(-layer.out[i] * layer.alphas[i])) / (1.0 + exp(-layer.out[i] * layer.alphas[i]));
            THRESHOLD:            if layer.out[i] > layer.alphas[i] then layer.out[i] := 1.0 else layer.out[i] := 0.0;
            LINEAR:               layer.out[i] := layer.out[i] * layer.alpha[i];
    end;

    result := layer.nodes;                                          //  Return the length of layer->out
end;
