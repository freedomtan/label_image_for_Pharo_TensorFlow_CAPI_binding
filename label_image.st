|IMAGE_SIZE INPUT_MEAN INPUT_STDDEV graph session scaledImage array2D b1 b2 b3 inputValues results labels kv|

IMAGE_SIZE := 224.
INPUT_MEAN := 128.0.
INPUT_STDDEV := 127.0.

scaledImage := (ImageReadWriter formFromFileNamed: '/tmp/grace_hopper.jpg') scaledIntoFormOfSize: IMAGE_SIZE.
b1 := scaledImage rgbaBitMasks at: 1.
b2 := scaledImage rgbaBitMasks at: 2.
b3 := scaledImage rgbaBitMasks at: 3.

array2D := Array2D new: IMAGE_SIZE.
1 to: IMAGE_SIZE do: [:x|
	 1 to: IMAGE_SIZE do: [:y|
		| p a |
		p := scaledImage pixelValueAt: x@y.
		a := Array 
			with: ((p & b1) bitShift: -16)
			with: ((p & b2) bitShift: -8)
			with: (p & b3)			
			.
		array2D at: y at: x put: a.
	 ]
].

graph := TF_Graph fromBinaryFileNamed: '/tmp/mobilenet_v1_1.0_224_quant_frozen.pb'.
session := TF_Session on: graph.
inputValues := TF_Tensor fromFloats: ((array2D - INPUT_MEAN) / INPUT_STDDEV) shape: (Array with: 1 with: IMAGE_SIZE with: IMAGE_SIZE with: 3).
results := ((session 
					runInputs: (Array with: ((graph operationAt: 1) output)) 
					values: (Array with: inputValues) 
					outputs: (Array with: ((graph operationAt: (graph operationsCount)) output))) at: 1) asNumbers at: 1.
					
labels := '/tmp/labels.txt' asFileReference contents lines.

kv := Array new: (results size).
1 to: (results size) do: [:i| kv at: i put: (Array with: (labels at: i) with: (results at: i))].
((kv sort: [:k1 :k2 | (k1 at: 2) > (k2 at: 2)]) first: 5) do: [:i|
	Transcript show: (i at: 1); show: ': '; show: (i at: 2); cr].
