	u:????t@u:????t@!u:????t@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu:????t@???$xAZ@1?Q*?	?k@AqN`:??I>"?D???rEagerKernelExecute 0*	??Q?E?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorZ??8?2@!?
?1%?X@)Z??8?2@1?
?1%?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf?ʉv??!????[???)f?ʉv??1????[???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????g??!zD?o????)??_>Y??1???.??:Preprocessing2F
Iterator::Modelp
+TT??!??|??)??1 ?n?1?۪?[???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?_̖??2@!??????X@)8fٓ??\?1P?6{O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI8?7g?0@@QddL??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???$xAZ@???$xAZ@!???$xAZ@      ??!       "	?Q*?	?k@?Q*?	?k@!?Q*?	?k@*      ??!       2	qN`:??qN`:??!qN`:??:	>"?D???>"?D???!>"?D???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q8?7g?0@@yddL??P@