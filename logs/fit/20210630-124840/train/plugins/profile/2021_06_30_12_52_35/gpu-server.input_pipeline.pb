	t??P?#y@t??P?#y@!t??P?#y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCt??P?#y@sJ@L¡c@1J???fn@A?ip[[??I????.???rEagerKernelExecute 0*	"??~???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??Gߤ)3@!?@de??X@)??Gߤ)3@1?@de??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?{eު???!0]B?m	??)?{eު???10]B?m	??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??_?????!_?Y3????)???????1?? UG??:Preprocessing2F
Iterator::ModelT??b???!???????)am???l?1?H??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap]?&?*3@!???9??X@)RH2?w?]?1????Z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIF??a9?C@Q??i??;N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sJ@L¡c@sJ@L¡c@!sJ@L¡c@      ??!       "	J???fn@J???fn@!J???fn@*      ??!       2	?ip[[???ip[[??!?ip[[??:	????.???????.???!????.???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qF??a9?C@y??i??;N@