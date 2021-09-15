unit NormalLayerUnit;

{Neural Network library, by Eric C. Joyce

 A normalizing layer applies the four learned parameters to its input.
  m = learned mean
  s = learned standard deviation
  g = learned coefficient
  b = learned constant

 input vec(x)    output vec(y)
   [ x1 ]     [ g*((x1 - m)/s)+b ]
   [ x2 ]     [ g*((x2 - m)/s)+b ]
   [ x3 ]     [ g*((x3 - m)/s)+b ]
   [ x4 ]     [ g*((x4 - m)/s)+b ]
   [ x5 ]     [ g*((x5 - m)/s)+b ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.}

interface

{**************************************************************************************************
 Constants  }

const
    LAYER_NAME_LEN = 32;                                            { Length of a Layer 'name' string }

//{$DEFINE __NORMAL_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    NormalLayer = record
        inputs: cardinal;                                           { Number of inputs }
        m: double;                                                  { Mu: the mean learned during training }
        s: double;                                                  { Sigma: the standard deviation learned during training }
        g: double;                                                  { The factor learned during training }
        b: double;                                                  { The constant learned during training }
        layerName: array [0..LAYER_NAME_LEN - 1] of char;
        out: array of double;
    end;

{**************************************************************************************************
 Prototypes  }

procedure setM_Normal(const m: double; var layer: NormalLayer);
procedure setS_Normal(const s: double; var layer: NormalLayer);
procedure setG_Normal(const g: double; var layer: NormalLayer);
procedure setB_Normal(const b: double; var layer: NormalLayer);
procedure setName_Normal(const n: array of char; var layer: NormalLayer);
procedure print_Normal(const layer: NormalLayer);
function run_Normal(const xvec: array of double; var layer: NormalLayer): cardinal;

implementation

{**************************************************************************************************
 Normalization-Layers  }

procedure setM_Normal(const m: double; var layer: NormalLayer);
begin
    layer.m := m;
end;

procedure setS_Normal(const s: double; var layer: NormalLayer);
begin
    layer.s := s;
end;

procedure setG_Normal(const g: double; var layer: NormalLayer);
begin
    layer.g := g;
end;

procedure setB_Normal(const b: double; var layer: NormalLayer);
begin
    layer.b := b;
end;

procedure setName_Normal(const n: array of char; var layer: NormalLayer);
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

procedure print_Normal(const layer: NormalLayer);
begin
    {$ifdef __NORMAL_DEBUG}
    writeln('print_Normal()');
    {$endif}

    writeln('Input Length = ', layer.inputs);
    writeln('Mean = ', layer.m);
    writeln('Std.dev = ', layer.s);
    writeln('Coefficient = ', layer.g);
    writeln('Constant = ', layer.b);
    writeln('');
end;

function run_Normal(const xvec: array of double; var layer: NormalLayer): cardinal;
var
    j: cardinal;
begin
    for j := 0 to layer.inputs - 1 do
        layer.out[j] := layer.g * ((xvec[j] - layer.m) / layer.s) + layer.b;
    result := layer.inputs;
end;