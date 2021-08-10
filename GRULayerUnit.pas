unit GRULayerUnit;

{Neural Network library, by Eric C. Joyce

 Model a GRU Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec(x)       weights Wz              weights Wr              weights Wh
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)
                 [ wz11 wz12 wz13 wz14 ] [ wr11 wr12 wr13 wr14 ] [ wh11 wh12 wh13 wh14 ]
                 [ wz21 wz22 wz23 wz24 ] [ wr21 wr22 wr23 wr24 ] [ wh21 wh22 wh23 wh24 ]
                 [ wz31 wz32 wz33 wz34 ] [ wr31 wr32 wr33 wr34 ] [ wh31 wh32 wh33 wh34 ]

                       weights Uz              weights Ur              weights Uh
                        (h by h)                (h by h)                (h by h)
                 [ uz11 uz12 uz13 ]      [ ur11 ur12 ur13 ]      [ uh11 uh12 uh13 ]
                 [ uz21 uz22 uz23 ]      [ ur21 ur22 ur23 ]      [ uh21 uh22 uh23 ]
                 [ uz31 uz32 uz33 ]      [ ur31 ur32 ur33 ]      [ uh31 uh32 uh33 ]

                     bias h-vec(bz)          bias h-vec(br)          bias h-vec(bh)
                 [ bz1 ]                 [ br1 ]                 [ bh1 ]
                 [ bz2 ]                 [ br2 ]                 [ bh2 ]
                 [ bz3 ]                 [ br3 ]                 [ bh3 ]

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

//{$DEFINE __GRU_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    GRULayer = record
        d: cardinal;                                                { Dimensionality of input vector }
        h: cardinal;                                                { Dimensionality of hidden state vector }
        cache: cardinal;                                            { The number of states to keep in memory: when 't' exceeds this, shift out. }

        t: cardinal;                                                { The time step }
                                                                    { W matrices are (h by d) }
        Wz: array of double;                                        { Update gate weights }
        Wr: array of double;                                        { Reset gate weights }
        Wh: array of double;                                        { Output gate weights }
                                                                    { U matrices are (h by h) }
        Uz: array of double;                                        { Recurrent connection update gate weights }
        Ur: array of double;                                        { Recurrent connection reset gate weights }
        Uh: array of double;                                        { Recurrent connection output gate weights }
                                                                    { Bias vectors are length h }
        bz: array of double;                                        { Update gate bias }
        br: array of double;                                        { Reset gate bias }
        bh: array of double;                                        { Output gate bias }

        H: array of double;                                         { Hidden state cache matrix (h by cache) }

        layerName: array [0..LAYER_NAME_LEN - 1] of char;
    end;

{**************************************************************************************************
 Prototypes  }

procedure setWz_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Wz weight matrix }
procedure setWr_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Wr weight matrix }
procedure setWh_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Wh weight matrix }
                                                                    { Set element [i, j] of Wz weight matrix }
procedure setWz_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
                                                                    { Set element [i, j] of Wr weight matrix }
procedure setWr_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
                                                                    { Set element [i, j] of Wh weight matrix }
procedure setWh_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
procedure setUz_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Uz weight matrix }
procedure setUr_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Ur weight matrix }
procedure setUh_GRU(const w: array of double; var layer: GRULayer); { Set entirety of Uh weight matrix }
                                                                    { Set element [i, j] of Uz weight matrix }
procedure setUz_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
                                                                    { Set element [i, j] of Ur weight matrix }
procedure setUr_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
                                                                    { Set element [i, j] of Uh weight matrix }
procedure setUh_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
procedure setbz_GRU(const w: array of double; var layer: GRULayer); { Set entirety of bz bias vector }
procedure setbr_GRU(const w: array of double; var layer: GRULayer); { Set entirety of br bias vector }
procedure setbh_GRU(const w: array of double; var layer: GRULayer); { Set entirety of bh bias vector }
                                                                    { Set i-th element of bz bias vector }
procedure setbz_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);
                                                                    { Set i-th element of br bias vector }
procedure setbr_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);
                                                                    { Set i-th element of bh bias vector }
procedure setbh_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);

procedure setName_GRU(const n: array of char; var layer: GRULayer);
procedure print_GRU(const layer: GRULayer);
function outputLen_GRU(const layer: GRULayer): cardinal;
function run_GRU(const xvec: array of double; var layer: GRULayer): cardinal;
procedure reset_GRU(var layer: GRULayer);

implementation

{**************************************************************************************************
 GRU-Layers  }

{ Set the entirety of the Wz matrix of the given layer using the given array }
procedure setWz_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wz[i] := w[i];
end;

{ Set the entirety of the Wr matrix of the given layer using the given array }
procedure setWr_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wr[i] := w[i];
end;

{ Set the entirety of the Wh matrix of the given layer using the given array }
procedure setWh_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.d * layer.h - 1 do
        layer.Wh[i] := w[i];
end;

{ Set column[i], row[j] of the given layer, Wz matrix }
procedure setWz_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setWz_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wz[j * layer.d + i] := w;
end;

{ Set column[i], row[j] of the given layer, Wr matrix }
procedure setWr_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setWr_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wr[j * layer.d + i] := w;
end;

{ Set column[i], row[j] of the given layer, Wh matrix }
procedure setWh_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setWh_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.d + j < layer.h * layer.d then
        layer.Wh[j * layer.d + i] := w;
end;

{ Set the entirety of the Uz matrix of the given layer using the given array }
procedure setUz_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Uz[i] := w[i];
end;

{ Set the entirety of the Ur matrix of the given layer using the given array }
procedure setUr_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Ur[i] := w[i];
end;

{ Set the entirety of the Uh matrix of the given layer using the given array }
procedure setUh_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h * layer.h - 1 do
        layer.Uh[i] := w[i];
end;

{ Set column[i], row[j] of the given layer, Uz matrix }
procedure setUz_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setUz_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Uz[j * layer.h + i] := w;
end;

{ Set column[i], row[j] of the given layer, Ur matrix }
procedure setUr_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setUr_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Ur[j * layer.h + i] := w;
end;

{ Set column[i], row[j] of the given layer, Uh matrix }
procedure setUh_ij_GRU(const w: double; const i, j: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setUh_ij_GRU(', w, ', ', i, ', ', j, ')');
    {$endif}

    if i * layer.h + j < layer.h * layer.h then
        layer.Uh[j * layer.h + i] := w;
end;

{ Set the entirety of the bz vector of the given layer using the given array }
procedure setbz_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bz[i] := w[i];
end;

{ Set the entirety of the br vector of the given layer using the given array }
procedure setbr_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.br[i] := w[i];
end;

{ Set the entirety of the bh vector of the given layer using the given array }
procedure setbh_GRU(const w: array of double; var layer: GRULayer);
var
    i: cardinal;
begin
    for i := 0 to layer.h - 1 do
        layer.bh[i] := w[i];
end;

{ Set element [i] of the given layer, bz vector }
procedure setbz_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setbz_i_GRU(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bz[i] := w;
end;

{ Set element [i] of the given layer, br vector }
procedure setbr_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setbr_i_GRU(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.br[i] := w;
end;

{ Set element [i] of the given layer, bh vector }
procedure setbh_i_GRU(const w: double; const i: cardinal; var layer: GRULayer);
begin
    {$ifdef __GRU_DEBUG}
    writeln('setbh_i_GRU(', w, ', ', i, ')');
    {$endif}

    if i < layer.h then
        layer.bh[i] := w;
end;

{ Set the name of the given GRU Layer }
procedure setName_GRU(const n: array of char; var layer: GRULayer);
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

{ Print the details of the given GRULayer 'layer' }
procedure print_GRU(const layer: GRULayer);
var
    i, j: cardinal;
begin
    {$ifdef __GRU_DEBUG}
    writeln('print_GRU()');
    {$endif}

    writeln('Input dimensionality d = ', layer.d);
    writeln('State dimensionality h = ', layer.h);
    writeln('State cache size       = ', layer.cache);

    writeln('Wz (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wz[i * layer.d + j]);
        writeln(' ]');
    end;
    writeln('Wr (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wr[i * layer.d + j]);
        writeln(' ]');
    end;
    writeln('Wh (', layer.h, ' x ', layer.d, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.d - 1 do
            write(' ', layer.Wh[i * layer.d + j]);
        writeln(' ]');
    end;

    writeln('');
    writeln('Uz (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Uz[i * layer.h + j]);
        writeln(' ]');
    end;
    writeln('Ur (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Ur[i * layer.h + j]);
        writeln(' ]');
    end;
    writeln('Uh (', layer.h, ' x ', layer.h, ')');
    for i := 0 to layer.h - 1 do
    begin
        write('[');
        for j := 0 to layer.h - 1 do
            write(' ', layer.Uh[i * layer.h + j]);
        writeln(' ]');
    end;

    writeln('');
    writeln('bz (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bz[i], ' ]');
    writeln('br (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.br[i], ' ]');
    writeln('bh (', layer.h, ' x 1)');
    for i := 0 to layer.h - 1 do
        writeln('[ ', layer.bh[i], ' ]');
end;

function outputLen_GRU(const layer: GRULayer): cardinal;
begin
    result := layer.h;
end;

{ Run the given input vector 'x' of length 'layer'.'d' through the GRULayer 'layer'.
  Output is stored internally in layer.H.
  Write to the 'layer'.'t'-th column and increment t.
  If 'layer'.'t' exceeds 'layer'.'cache', shift everything down. }
function run_GRU(const x: array of double; var layer: GRULayer): cardinal;
var
    n, m: cardinal;
    z: array of double;                                             //  These act as accumulator vectors
    r: array of double;
    h: array of double;
    hprime: array of double;                                        //  Intermediate Hadamard product r * ht_1
    ht_1: array of double;                                          //  Time t - 1

    t_1: cardinal;                                                  //  Where we READ FROM
    t: cardinal;                                                    //  Where we WRITE TO
                                                                    //  layer.t increases indefinitely
begin
    {$ifdef __GRU_DEBUG}
    writeln('run_GRU(', layer.h, ')');
    {$endif}

    SetLength(z, layer.h);                                          //  Allocate vec z
    SetLength(r, layer.h);                                          //  Allocate vec r
    SetLength(h, layer.h);                                          //  Allocate vec h
    SetLength(hprime, layer.h);                                     //  Allocate vec h'

    SetLength(ht_1, layer.h);                                       //  Allocate vec ht_1

    if layer.t = 0 then                                             //  Timestep layer->t = 0 uses the zero-vectors for t - 1
    begin
        t_1 := 0;
        t := 0;
        for n := 0 to layer.h - 1 do                                //  Write zeroes to h(t-1) and c(t-1)
            ht_1[n] := 0.0;
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
            ht_1[n] := layer.H[ n * layer.cache + t_1 ];
    end;

    for n := 0 to layer.h - 1 do                                    //  Write biases to vectors
    begin
        z[n] := layer.bz[n];
        r[n] := layer.br[n];
        h[n] := 0.0;                                                //  Blank out h
    end;
                                                                    //  Add Uz dot ht_1 to z
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Uz, layer.h, ht_1, 1, 1.0, z, 1);
                                                                    //  Add Ur dot ht_1 to r
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Ur, layer.h, ht_1, 1, 1.0, r, 1);

    if layer.d = 1 then                                             //  Special case when input has dimension 1
    begin
        for n := 0 to layer.h - 1 do
        begin
            z[n] += layer.Wz[n] * x[0];                             //  Add Wz dot x to z
            r[n] += layer.Wr[n] * x[0];                             //  Add Wr dot x to r
        end;
    end
    else
    begin
                                                                    //  Add Wz dot x to z
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wz, layer.h, x, 1, 1.0, z, 1);
                                                                    //  Add Wr dot x to r
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wr, layer.h, x, 1, 1.0, r, 1);
    end;
    for n := 0 to layer.h - 1 do                                    //  Apply sigmoid function to z and r vectors
    begin
        z[n] := 1.0 / (1.0 + exp(-z[n]));                           //  z = sig(Wz.x + Uz.ht_1 + bz)
        r[n] := 1.0 / (1.0 + exp(-r[n]));                           //  r = sig(Wr.x + Ur.ht_1 + br)
    end;

    for n := 0 to layer.h - 1 do                                    //  h' = r * ht_1
      hprime[n] := r[n] * ht_1[n];
                                                                    //  Set h = Uh.h' = Uh.(r * ht_1)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.h, 1.0, layer.Uh, layer.h, hprime, 1, 1.0, h, 1);
    if layer.d = 1 then                                             //  Add Wh dot x to h
    begin
        for n := 0 to layer.h - 1 do
            h[n] += layer.Wh[n] * x[0];                             //  Add Wh dot x to h
    end
    else
        cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.h, layer.d, 1.0, layer.Wh, layer.h, x, 1, 1.0, h, 1);
    for n := 0 to layer.h - 1 do                                    //  Add bias to h vector
        h[n] += layer.bh[n];

    //  Now h = Wh.x + Uh.(r * ht_1) + bh

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

    for n := 0 to layer.h - 1 do                                    //  h = z*ht_1 + (1-z)*tanh(h)
        layer.H[ n * layer.cache + t ] := z[n] * ht_1[n] + (1.0 - z[n]) * ((2.0 / (1.0 + exp(-2.0 * h[n]))) - 1.0);

    SetLength(z, 0);                                                //  Clean up, go home
    SetLength(r, 0);
    SetLength(h, 0);
    SetLength(hprime, 0);
    SetLength(ht_1, 0);

    layer.t += 1;                                                   //  Increment time step

    result := layer.h;                                              //  Return the size of the state
end;

procedure reset_GRU(var layer: GRULayer);
var
    i: cardinal;
begin
    layer.t := 0;                                                   //  Reset time step
    for i := 0 to layer.h * layer.cache - 1 do                      //  Blank out H
        layer.H[i] := 0.0;
end;
