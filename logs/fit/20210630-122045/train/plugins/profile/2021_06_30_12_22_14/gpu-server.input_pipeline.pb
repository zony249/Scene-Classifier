	???x@???x@!???x@	???v*z????v*z?!???v*z?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???x@???<?c@1????spn@A5^?I??I?ݓ??Z??Ym?i?*???rEagerKernelExecute 0*	??nk??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorEdX?iC@!Ce???X@)EdX?iC@1Ce???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch1??B?ʐ?!Q???[???)1??B?ʐ?1Q???[???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`?o`r???!???T?h??)??~?{??1?????5??:Preprocessing2F
Iterator::Model膦?????!?k6??ҷ?)z?m?(n?1??A%R??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??^iC@!e??H?X@)??ډ?`?1?{???Ru?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???v*z?I?μXC@Q?܀ﮦN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???<?c@???<?c@!???<?c@      ??!       "	????spn@????spn@!????spn@*      ??!       2	5^?I??5^?I??!5^?I??:	?ݓ??Z???ݓ??Z??!?ݓ??Z??B      ??!       J	m?i?*???m?i?*???!m?i?*???R      ??!       Z	m?i?*???m?i?*???!m?i?*???b      ??!       JGPUY???v*z?b q?μXC@y?܀ﮦN@