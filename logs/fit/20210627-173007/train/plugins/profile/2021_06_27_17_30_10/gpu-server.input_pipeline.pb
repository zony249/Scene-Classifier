	Ǟ=???q@Ǟ=???q@!Ǟ=???q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCǞ=???q@0??9:X@1??C6?f@A????ģ?I?熦?4??rEagerKernelExecute 0*	?S??s??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???D@@!O9L??X@)???D@@1O9L??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?wD???!????Z??)?wD???1????Z??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?t"?T3??!?????x??)?????1???{w<??:Preprocessing2F
Iterator::Model(G?`Ƥ?!?6Po????)?(??0i?1???TU??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapP??@D@@!?+??X@)?E??U?1q????p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI*":???A@Q??b`&:P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0??9:X@0??9:X@!0??9:X@      ??!       "	??C6?f@??C6?f@!??C6?f@*      ??!       2	????ģ?????ģ?!????ģ?:	?熦?4???熦?4??!?熦?4??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q*":???A@y??b`&:P@