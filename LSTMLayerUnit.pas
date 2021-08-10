unit LSTMLayerUnit;

{Neural Network library, by Eric C. Joyce

 Model an LSTM Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec(x)       weights Wi              weights Wo              weights Wf              weights Wc
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)                (h by d)
                 [ wi11 wi12 wi13 wi14 ] [ wo11 wo12 wo13 wo14 ] [ wf11 wf12 wf13 wf14 ] [ wc11 wc12 wc13 wc14 ]
                 [ wi21 wi22 wi23 wi24 ] [ wo21 wo22 wo23 wo24 ] [ wf21 wf22 wf23 wf24 ] [ wc21 wc22 wc23 wc24 ]
                 [ wi31 wi32 wi33 wi34 ] [ wo31 wo32 wo33 wo34 ] [ wf31 wf32 wf33 wf34 ] [ wc31 wc32 wc33 wc34 ]

                       weights Ui              weights Uo              weights Uf              weights Uc
                        (h by h)                (h by h)                (h by h)                (h by h)
                 [ ui11 ui12 ui13 ]      [ uo11 uo12 uo13 ]      [ uf11 uf12 uf13 ]      [ uc11 uc12 uc13 ]
                 [ ui21 ui22 ui23 ]      [ uo21 uo22 uo23 ]      [ uf21 uf22 uf23 ]      [ uc21 uc22 uc23 ]
                 [ ui31 ui32 ui33 ]      [ uo31 uo32 uo33 ]      [ uf31 uf32 uf33 ]      [ uc31 uc32 uc33 ]

                     bias h-vec(bi)          bias h-vec(bo)          bias h-vec(bf)          bias h-vec(bc)
                 [ bi1 ]                 [ bo1 ]                 [ bf1 ]                 [ bc1 ]
                 [ bi2 ]                 [ bo2 ]                 [ bf2 ]                 [ bc2 ]
                 [ bi3 ]                 [ bo3 ]                 [ bf3 ]                 [ bc3 ]

         H state cache (times 1, 2, 3, 4 = columns 0, 1, 2, 3)
        (h by cache)
 [ H11 H12 H13 H14 ]
 [ H21 H22 H23 H24 ]
 [ H31 H32 H33 H34 ]

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

//{$DEFINE __LSTM_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    LSTMLayer = record
        d: cardinal;                                                { Dimensionality of input vector }
        h: cardinal;                                                { Dimensionality of hidden state vector }
        cache: cardinal;                                            { The number of states to keep in memory: when 't' exceeds this, shift out. }

        t: cardinal;                                                { The time step }
                                                                    { W matrices are (h by d) }
        Wi: array of double;                                        { Input gate weights }
        Wo: array of double;                                        { Output gate weights }
        Wf: array of double;                                        { Forget gate weights }
        Wc: array of double;                                        { Memory cell weights }
                                                                    { U matrices are (h by h) }
        Ui: array of double;                                        { Recurrent connection input gate weights }
        Uo: array of double;                                        { Recurrent connection output gate weights }
        Uf: array of double;                                        { Recurrent connection forget gate weights }
        Uc: array of double;                                        { Recurrent connection memory cell weights }
                                                                    { Bias vectors are length h }
        bi: array of double;                                        { Input gate bias }
        bo: array of double;                                        { Output gate bias }
        bf: array of double;                                        { Forget gate bias }
        bc: array of double;                                        { Memory cell bias }

        c: array of double;                                         { Cell state vector, length h }
        H: array of double;                                         { Hidden state cache matrix (h by cache) }

        layerName: array [0..LAYER_NAME_LEN - 1] of char;
    end;

{**************************************************************************************************
 Prototypes  }

                                                                    { Set entirety of Wi weight matrix }
procedure setWi_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Wo weight matrix }
procedure setWo_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Wf weight matrix }
procedure setWf_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Wc weight matrix }
procedure setWc_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Wi weight matrix }
procedure setWi_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Wo weight matrix }
procedure setWo_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Wf weight matrix }
procedure setWf_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Wc weight matrix }
procedure setWc_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set entirety of Ui weight matrix }
procedure setUi_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Uo weight matrix }
procedure setUo_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Uf weight matrix }
procedure setUf_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of Uc weight matrix }
procedure setUc_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Ui weight matrix }
procedure setUi_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Uo weight matrix }
procedure setUo_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Uf weight matrix }
procedure setUf_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set element [i, j] of Uc weight matrix }
procedure setUc_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
                                                                    { Set entirety of bi bias vector }
procedure setbi_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of bo bias vector }
procedure setbo_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of bf bias vector }
procedure setbf_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set entirety of bc bias vector }
procedure setbc_LSTM(const w: array of double; var layer: LSTMLayer);
                                                                    { Set i-th element of bi bias vector }
procedure setbi_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
                                                                    { Set i-th element of bo bias vector }
procedure setbo_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
                                                                    { Set i-th element of bf bias vector }
procedure setbf_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
                                                                    { Set i-th element of bc bias vector }
procedure setbc_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
procedure setName_LSTM(const n: array of char; var layer: LSTMLayer);
procedure print_LSTM(const layer: LSTMLayer);
function outputLen_LSTM(const layer: LSTMLayer): cardinal;
function run_LSTM(const xvec: array of double; var layer: LSTMLayer): cardinal;
procedure reset_LSTM(var layer: LSTMLayer);

implementation

{**************************************************************************************************
 LSTM-Layers  }

{ Set the entirety of the Wi matrix of the given layer using the given array }
procedure setWi_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wi[i] := w[i];
end;

{ Set the entirety of the Wo matrix of the given layer using the given array }
procedure setWo_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wo[i] := w[i];
end;

{ Set the entirety of the Wf matrix of the given layer using the given array }
procedure setWf_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wf[i] := w[i];
end;

{ Set the entirety of the Wc matrix of the given layer using the given array }
procedure setWc_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wc[i] := w[i];
end;

{ Set column[i], row[j] of the given layer, Wi matrix }
procedure setWi_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setWi_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wi[j * layer.d + i] := w;
end;

{ Set column[i], row[j] of the given layer, Wo matrix }
procedure setWo_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setWo_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wo[j * layer.d + i] := w;
end;

{ Set column[i], row[j] of the given layer, Wf matrix }
procedure setWf_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setWf_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wf[j * layer.d + i] := w;
end;

{ Set column[i], row[j] of the given layer, Wc matrix }
procedure setWc_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setWc_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wc[j * layer.d + i] := w;
end;

{ Set the entirety of the Ui matrix of the given layer using the given array }
procedure setUi_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Ui[i] := w[i];
end;

{ Set the entirety of the Uo matrix of the given layer using the given array }
procedure setUo_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Uo[i] := w[i];
end;

{ Set the entirety of the Uf matrix of the given layer using the given array }
procedure setUf_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Uf[i] := w[i];
end;

{ Set the entirety of the Uc matrix of the given layer using the given array }
procedure setUc_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Uc[i] := w[i];
end;

{ Set column[i], row[j] of the given layer, Ui matrix }
procedure setUi_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setUi_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Ui[j * layer.h + i] := w;
end;

{ Set column[i], row[j] of the given layer, Uo matrix }
procedure setUo_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setUo_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Uo[j * layer.h + i] := w;
end;

{ Set column[i], row[j] of the given layer, Uf matrix }
procedure setUf_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setUf_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Uf[j * layer.h + i] := w;
end;

{ Set column[i], row[j] of the given layer, Uc matrix }
procedure setUc_ij_LSTM(const w: double; const i, j: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setUc_ij_LSTM(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Uc[j * layer.h + i] := w;
end;

{ Set the entirety of the bi vector of the given layer using the given array }
procedure setbi_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bi[i] := w[i];
end;

{ Set the entirety of the bo vector of the given layer using the given array }
procedure setbo_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bo[i] := w[i];
end;

{ Set the entirety of the bf vector of the given layer using the given array }
procedure setbf_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bf[i] := w[i];
end;

{ Set the entirety of the bc vector of the given layer using the given array }
procedure setbc_LSTM(const w: array of double; var layer: LSTMLayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bc[i] := w[i];
end;

{ Set element [i] of the given layer, bi vector }
procedure setbi_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setbi_i_LSTM(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bi[i] := w;
end;

{ Set element [i] of the given layer, bo vector }
procedure setbo_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setbo_i_LSTM(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bo[i] := w;
end;

{ Set element [i] of the given layer, bf vector }
procedure setbf_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setbf_i_LSTM(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bf[i] := w;
end;

{ Set element [i] of the given layer, bc vector }
procedure setbc_i_LSTM(const w: double; const i: cardinal; var layer: LSTMLayer);
begin
    {$ifdef __LSTM_DEBUG}
    writeln('setbc_i_LSTM(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bc[i] := w;
end;

{ Set the name of the given LSTM Layer }
procedure setName_LSTM(const n: array of char; var layer: LSTMLayer);
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

{ Print the details of the given LSTMLayer 'layer' }
procedure print_LSTM(const layer: LSTMLayer);
var
    i, j: cardinal;
begin
    {$ifdef __LSTM_DEBUG}
    writeln('print_LSTM()');
    {$endif}

    writeln('Input dimensionality d = ', layer.d);
    writeln('State dimensionality h = ', layer.h);
    writeln('State cache size       = ', layer.cache);

    writeln('Wi (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wi[i * layer.d + j]);
        writeln(' ]');
    end;
    writeln('Wf (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wf[i * layer.d + j]);
        writeln(' ]');
    end;
    writeln('Wc (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wc[i * layer.d + j]);
        writeln(' ]');
    end;
    writeln('Wo (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wo[i * layer.d + j]);
        writeln(' ]');
    end;

    writeln('');
    writeln('Ui (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Ui[i * layer.h + j]);
        writeln(' ]');
    end;
    writeln('Uf (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Uf[i * layer.h + j]);
        writeln(' ]');
    end;
    writeln('Uc (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Uc[i * layer.h + j]);
        writeln(' ]');
    end;
    writeln('Uo (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Uo[i * layer.h + j]);
        writeln(' ]');
    end;

    writeln('');
    writeln('bi (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bi[i], ' ]');
    writeln('bf (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bf[i], ' ]');
    writeln('bc (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bc[i], ' ]');
    writeln('bo (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bo[i], ' ]');
end;

function outputLen_LSTM(const layer: LSTMLayer): cardinal;
begin
    result := layer.h;
end;

{ Run the given input vector 'x' of length 'layer'.'d' through the LSTMLayer 'layer'.
  Output is stored internally in layer.H.
  Write to the 'layer'.'t'-th column and increment t.
  If 'layer'.'t' exceeds 'layer'.'cache', shift everything down. }
function run_LSTM(const x: array of double; var layer: LSTMLayer): cardinal;
var
    n, m: cardinal;
    i: array of double;                                             //  These act as accumulator vectors
    f: array of double;
    c: array of double;                                             //  Time t (c-tilde in the equation)
    o: array of double;
    ct_1: array of double;                                          //  Time t - 1
    ht_1: array of double;                                          //  Time t - 1

    t_1: cardinal;                                                  //  Where we READ FROM
    t: cardinal;                                                    //  Where we WRITE TO
                                                                    //  layer.t increases indefinitely
begin
    {$ifdef __LSTM_DEBUG}
    writeln('run_LSTM(', layer.h, ')');
    {$endif}

    SetLength(i, layer.h);                                          //  Allocate vec i
    SetLength(f, layer.h);                                          //  Allocate vec f
    SetLength(c, layer.h);                                          //  Allocate vec c
    SetLength(o, layer.h);                                          //  Allocate vec o

    SetLength(ht_1, layer.h);                                       //  Allocate vec ht_1
    SetLength(ct_1, layer.h);                                       //  Allocate vec ct_1

    if layer.t = 0 then                                             //  Timestep layer->t = 0 uses the zero-vectors for t - 1
    begin
        t_1 := 0;
        t := 0;
        for n := 0 to layer.h - 1 do                                //  Write zeroes to h(t-1) and c(t-1)
        begin
            ht_1[n] := 0.0;
            ct_1[n] := 0.0;
        end;
    end
    else                                                            //  Timestep t > 0 uses the previous state
    begin                                                           //  Consider that we may have shifted states
        if layer.t >= layer.cache then                              //  out of the matrix
        begin
            t_1 := layer.cache - 1;                                 //  Read from the rightmost column
                                                                    //  (then shift everything left)
            t := layer.cache - 1;                                   //  Write to the rightmost column
        end
        else                                                        //  We've not yet maxed out cache
        begin
            t_1 := layer.t - 1;                                     //  Read from the previous column
            t := layer.t;                                           //  Write to the targeted column
        end;
        for n := 0 to layer.h - 1 do
        begin
            ht_1[n] := layer.H[ n * layer.cache + t_1 ];
            ct_1[n] := layer.c[n];
        end;
    end;

    //  f_t  = sig_g(W_f * x_t + U_f * h_{t-1} + b_f)
    //  i_t  = sig_g(W_i * x_t + U_i * h_{t-1} + b_i)
    //  o_t  = sig_g(W_o * x_t + U_o * h_{t-1} + b_o)
    //  c~_t = sig_c(W_c * x_t + U_c * h_{t-1} + b_c)
    //  c_t  = f_t Hadamard c_{t-1} + i_t Hadamard c~_t
    //  h_t  = o_t Hadamard sig_h(c_t)

    for n := 0 to layer.h - 1 do                                    //  Write biases to vectors
    begin
        i[n] := layer.bi[n];
        f[n] := layer.bf[n];
        c[n] := layer.bc[n];
        o[n] := layer.bo[n];
    end;
                                                                    //  Add Ui dot ht_1 to i (i = U_i * h_{t-1} + b_i)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Ui, layer.h, ht_1, 1, 1.0, i, 1);
                                                                    //  Add Uf dot ht_1 to f (f = U_f * h_{t-1} + b_f)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Uf, layer.h, ht_1, 1, 1.0, f, 1);
                                                                    //  Add Uc dot ht_1 to c (c = U_c * h_{t-1} + b_c)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Uc, layer.h, ht_1, 1, 1.0, c, 1);
                                                                    //  Add Uo dot ht_1 to o (o = U_o * h_{t-1} + b_o)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Uo, layer.h, ht_1, 1, 1.0, o, 1);

    if layer.d = 1 then                                             //  Special case when input has dimension 1
    begin
        for n := 0 to layer.h - 1 do
        begin
            i[n] += layer.Wi[n] * x[0];                             //  Add Wi dot x to i (i = W_i * x_t + U_i * h_{t-1} + b_i)
            f[n] += layer.Wf[n] * x[0];                             //  Add Wf dot x to f (f = W_f * x_t + U_f * h_{t-1} + b_f)
            c[n] += layer.Wc[n] * x[0];                             //  Add Wc dot x to c (c = W_c * x_t + U_c * h_{t-1} + b_c)
            o[n] += layer.Wo[n] * x[0];                             //  Add Wo dot x to o (o = W_o * x_t + U_o * h_{t-1} + b_o)
        end;
    end
    else
    begin
                                                                    //  Add Wi dot x to i (i = W_i * x_t + U_i * h_{t-1} + b_i)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wi, layer.h, x, 1, 1.0, i, 1);
                                                                    //  Add Wf dot x to f (f = W_f * x_t + U_f * h_{t-1} + b_f)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wf, layer.h, x, 1, 1.0, f, 1);
                                                                    //  Add Wc dot x to c (c = W_c * x_t + U_c * h_{t-1} + b_c)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wc, layer.h, x, 1, 1.0, c, 1);
                                                                    //  Add Wo dot x to o (o = W_o * x_t + U_o * h_{t-1} + b_o)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wo, layer.h, x, 1, 1.0, o, 1);
    end;

    //  We have allocated h-by-cache space for 'H', but the structure and routine should not crash if
    //  we write more than 'cache' states. Shift everything down one column and write to the end.
    if layer.t >= layer.cache then
    begin
        for m := 1 to layer.cache - 1 do                            //  Shift down
        begin
            for n := 0 to layer.h - 1 do
                layer.H[n * layer.cache + m - 1] := layer.H[n * layer.cache + m];
        end;
    end;

    for n := 0 to layer.h - 1 do
    begin
        i[n] := 1.0 / (1.0 + exp(-i[n]));                           //  i = sig(W_i*x_t + U_i*h_{t-1} + b_i)
        f[n] := 1.0 / (1.0 + exp(-f[n]));                           //  f = sig(W_f*x_t + U_f*h_{t-1} + b_f)
                                                                    //  c = f_t Hadamard c_{t-1} + i_t Hadamard tanh(W_c*x_t + U_c*h_{t-1} + b_c)
        layer.c[n] := f[n] * ct_1[n] + i[n] * ((2.0 / (1.0 + exp(-2.0 * c[n]))) - 1.0);
        o[n] := 1.0 / (1.0 + exp(-o[n]));                           //  o = sig(W_o*x_t + U_o*h_{t-1} + b_o)
                                                                    //  h = o_t Hadamard tanh(c_t)
        layer.H[ n * layer.cache + t ] := o[n] * ((2.0 / (1.0 + exp(-2.0 * layer.c[n]))) - 1.0);
    end;

    SetLength(i, 0);                                                //  Clean up, go home
    SetLength(f, 0);
    SetLength(c, 0);
    SetLength(o, 0);
    SetLength(ct_1, 0);
    SetLength(ht_1, 0);

    layer.t += 1;                                                   //  Increment time step

    result := layer.h;                                              //  Return the size of the state
end;

procedure reset_LSTM(var layer: LSTMLayer);
var
    i: cardinal;
begin
    layer.t := 0;                                                   //  Reset time step
    for i := 0 to layer.h - 1 do                                    //  Blank out c
        layer.c[i] := 0.0;
    for i := 0 to layer.h * layer.cache - 1 do                      //  Blank out H
        layer.H[i] := 0.0;
end;
