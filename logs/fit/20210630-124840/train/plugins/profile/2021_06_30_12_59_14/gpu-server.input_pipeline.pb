	?2?}1y@?2?}1y@!?2?}1y@	???^??????^???!???^???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?2?}1y@?m3??c@1??8<<n@A??mT???I?q7?????Y,?S???rEagerKernelExecute 0*	'1???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??L0??=@!-7??X@)??L0??=@1-7??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch	?????!?C?ȏ???)	?????1?C?ȏ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?}?Az???!#?w^????)*?Z^?ކ?1??K??7??:Preprocessing2F
Iterator::Model??9??q??!?S?????) ?8?@dq?1w??^z:??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap+5{??=@!:k?B?X@)6??\^?1?7ނy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???^???I??B?C@QL?x?dN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m3??c@?m3??c@!?m3??c@      ??!       "	??8<<n@??8<<n@!??8<<n@*      ??!       2	??mT?????mT???!??mT???:	?q7??????q7?????!?q7?????B      ??!       J	,?S???,?S???!,?S???R      ??!       Z	,?S???,?S???!,?S???b      ??!       JGPUY???^???b q??B?C@yL?x?dN@