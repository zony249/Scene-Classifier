	?Xz$y@?Xz$y@!?Xz$y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Xz$y@?a????c@1-??\nOn@A??9?٘?I??U-?h??rEagerKernelExecute 0*	H?z^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?k????5@!p????X@)?k????5@1p????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????#ӑ?!?D?`㨴?)????#ӑ?1?D?`㨴?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismM?*?????!??>??3??)}%?????1ʈ?A|??:Preprocessing2F
Iterator::Model??ek}???!???4??)?uʣk?1ۧ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?Z|
??5@!@???e?X@)-?}?a?1;u?b桄?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIN?\c??C@Q???t#N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?a????c@?a????c@!?a????c@      ??!       "	-??\nOn@-??\nOn@!-??\nOn@*      ??!       2	??9?٘???9?٘?!??9?٘?:	??U-?h????U-?h??!??U-?h??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qN?\c??C@y???t#N@