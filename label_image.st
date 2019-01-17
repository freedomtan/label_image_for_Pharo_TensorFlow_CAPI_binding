|buffer graph session scaledImage array2D bitmasks p b1 b2 b3 a inputValues results labels kv|

scaledImage := (ImageReadWriter formFromFileNamed: '/tmp/grace_hopper.jpg') scaledIntoFormOfSize: 224.
bitmasks := scaledImage rgbaBitMasks.
b1 := bitmasks at: 1.
b2 := bitmasks at: 2.
b3 := bitmasks at: 3.
array2D := Array2D new: 224.
1 to: 224 do: [:x|
	 1 to: 224 do: [:y|
		p := scaledImage pixelValueAt: x@y.
		a := Array 
			with: ((p & b1) bitShift: -16)
			with: ((p & b2) bitShift: -8)
			with: (p & b3)			
			.
		array2D at: y at: x put: a.
	 ]
].

buffer := TF_Buffer fromBinaryFileNamed: '/tmp/mobilenet_v1_1.0_224_quant_frozen.pb'.
graph := [TF_Graph create import: buffer] ensure: [buffer delete].
session := TF_Session on: graph.
inputValues := TF_Tensor fromFloats: ((array2D - 128)/127.0) shape: (Array with: 1 with: 224 with: 224 with: 3).
results := ((session 
					runInputs: (Array with: ((graph operationAt: 1) output)) 
					values: (Array with: inputValues) 
					outputs: (Array with: ((graph operationAt: (graph operationsCount)) output))) at: 1) asNumbers at: 1.
					
labels := '/tmp/labels.txt' asFileReference contents lines.

kv := Array new: 1001.
1 to: 1001 do: [:i| kv at: i put: (Array with: (labels at: i) with: (results at: i))].
((kv sort: [:k1 :k2 | (k1 at: 2) > (k2 at: 2)]) first: 5) do: [:i|
	Transcript show: (i at: 1); show: ': '; show: (i at: 2); cr].
