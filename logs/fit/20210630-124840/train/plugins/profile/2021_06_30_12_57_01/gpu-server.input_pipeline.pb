	????(Fy@????(Fy@!????(Fy@	?-^??????-^?????!?-^?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????(Fy@?v?>X?c@1:毐In@A}\*????I<1??P?@YJ?????rEagerKernelExecute 0*	@5^?Q?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????ɶ=@!m?6j?X@)????ɶ=@1m?6j?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?@?"??!??P??j??)?@?"??1??P??j??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismuYLl>???!?"??;???)?1??8*??1W?&?w??:Preprocessing2F
Iterator::Modelu???l??!9!??۔??)???mRq?1??GJ ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap*???O?=@!7B???X@)???$??`?1z??N?$|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?-^?????IjE?d?D@Q?D??}?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?>X?c@?v?>X?c@!?v?>X?c@      ??!       "	:毐In@:毐In@!:毐In@*      ??!       2	}\*????}\*????!}\*????:	<1??P?@<1??P?@!<1??P?@B      ??!       J	J?????J?????!J?????R      ??!       Z	J?????J?????!J?????b      ??!       JGPUY?-^?????b qjE?d?D@y?D??}?M@