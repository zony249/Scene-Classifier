	8?k???x@8?k???x@!8?k???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8?k???x@?q4GV?b@1?L1ATn@Aɓ?k&ߜ?Iv?|?H?@rEagerKernelExecute 0*	?rh?U>?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorR?Q???@!X?~?X@)R?Q???@1X?~?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?	Q???!"D??*??)?	Q???1"D??*??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?f?ba???!!0T?K??)C?Գ ???1!Y???l??:Preprocessing2F
Iterator::ModelA?+????!jP???)cAJh?1??"????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapl#??f??@!???9??X@)ҏ?S??[?1?{#v??u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??ȝ?kC@QN7b ?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q4GV?b@?q4GV?b@!?q4GV?b@      ??!       "	?L1ATn@?L1ATn@!?L1ATn@*      ??!       2	ɓ?k&ߜ?ɓ?k&ߜ?!ɓ?k&ߜ?:	v?|?H?@v?|?H?@!v?|?H?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??ȝ?kC@yN7b ?N@