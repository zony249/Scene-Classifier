	I?Ǵ6y@I?Ǵ6y@!I?Ǵ6y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCI?Ǵ6y@?}?mAc@1???d?n@A˅ʿ?W??I C??@rEagerKernelExecute 0*	ˡE??3?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?T????A@!??bo9?X@)?T????A@1??bo9?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?k?,	P??!(?5??h??)?k?,	P??1(?5??h??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?(	?????!??c?>d??)tys?V{??1??P?_??:Preprocessing2F
Iterator::Model=??tZ???!?????$??)?~?o?1????^??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapY?E???A@!?B_Ŷ?X@)?f??f?1??U?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ?*???C@Q?i?9	{N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}?mAc@?}?mAc@!?}?mAc@      ??!       "	???d?n@???d?n@!???d?n@*      ??!       2	˅ʿ?W??˅ʿ?W??!˅ʿ?W??:	 C??@ C??@! C??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?*???C@y?i?9	{N@