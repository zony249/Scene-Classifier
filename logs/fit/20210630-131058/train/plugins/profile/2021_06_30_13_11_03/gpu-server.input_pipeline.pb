	FD1yy@FD1yy@!FD1yy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCFD1yy@|???ǟc@1??b?$n@A[??K????Ink?K??rEagerKernelExecute 0*	Y9?Ⱦ^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator9EGr?=@!??_0?X@)9EGr?=@1??_0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?͌~4???!?q?????)?͌~4???1?q?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm?/?r??!?*?W?3??)'?E'K???1"rwX`??:Preprocessing2F
Iterator::Modeleު?PM??!p?Yǽ?)y=??p?1????????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap7??nf=@!???)??X@)?t?? ?[?1,?#}rw?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI|j?P_?C@Q??`??#N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|???ǟc@|???ǟc@!|???ǟc@      ??!       "	??b?$n@??b?$n@!??b?$n@*      ??!       2	[??K????[??K????![??K????:	nk?K??nk?K??!nk?K??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|j?P_?C@y??`??#N@