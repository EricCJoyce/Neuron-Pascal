unit AccumLayerUnit;

{Neural Network library, by Eric C. Joyce

 Note that this file does NOT seed the randomizer. That should be done by the parent program.}

interface

{**************************************************************************************************
 Constants  }

const
    LAYER_NAME_LEN      = 32;                                       { Length of a Layer 'name' string }

//{$DEFINE __ACCUM_DEBUG 1}

{**************************************************************************************************
 Typedefs  }

type
    AccumLayer = record
        inputs: cardinal;                                           { Number of inputs--NOT COUNTING the added bias-1 }
        layerName: array [0..LAYER_NAME_LEN - 1] of char;
        out: array of double;
    end;

{**************************************************************************************************
 Prototypes  }

procedure setName_Dense(const n: array of char; var layer: AccumLayer);

implementation

{**************************************************************************************************
 Accumulator-Layers  }

{ Set the name of the given Accumulator Layer }
procedure setName_Dense(const n: array of char; var layer: AccumLayer);
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
