	?k??\z@?k??\z@!?k??\z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?k??\z@??̦f@1?-?en@A?I?U؜?I????U??rEagerKernelExecute 0*	B`?к??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???`?)<@!?????X@)???`?)<@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?)X?l:??!??$|)??)?)X?l:??1??$|)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???N??!E??Mˢ??)B]¡??1???w???:Preprocessing2F
Iterator::Model?r/0+??!DJZ?^H??)Hk:!tp?1????,??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?uoEb*<@!m?\?m?X@)臭???c?1\??
c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIM?F??,E@Q?t?U?L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??̦f@??̦f@!??̦f@      ??!       "	?-?en@?-?en@!?-?en@*      ??!       2	?I?U؜??I?U؜?!?I?U؜?:	????U??????U??!????U??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qM?F??,E@y?t?U?L@