	????n@????n@!????n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????n@O??:7???1Ҩ??6n@A~?[?~lr?I??????@rEagerKernelExecute 0*	<?O??6?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorzȔA?=@!?@?w??X@)zȔA?=@1?@?w??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?y ?????!??~]???)?y ?????1??~]???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??	Q??!ah+?U??)/?????1n?Q?o???:Preprocessing2F
Iterator::Model??x"????!Ĝ?/2U??)??9?l?1???!????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?????=@!?t???X@)z7eZ?1?x??v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI????????QF=H?]?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O??:7???O??:7???!O??:7???      ??!       "	Ҩ??6n@Ҩ??6n@!Ҩ??6n@*      ??!       2	~?[?~lr?~?[?~lr?!~?[?~lr?:	??????@??????@!??????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????????yF=H?]?X@