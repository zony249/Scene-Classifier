	????Vz@????Vz@!????Vz@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????Vz@??:??Ge@1?O=R o@Alv???/??I?ЕT@rEagerKernelExecute 0*	?G?r]?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??唀 =@!?z?}W?X@)??唀 =@1?z?}W?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchԀAҧU??!??j????)ԀAҧU??1??j????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?|ԛQ??!?N?ڻ???)y?ՏM??1?+?`с??:Preprocessing2F
Iterator::Model?g\8???!%l?????)qW?"?r?1?d????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?#0? =@!?I????X@)ҏ?S??[?1????x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??ޭ?tD@Qj{!RA?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??:??Ge@??:??Ge@!??:??Ge@      ??!       "	?O=R o@?O=R o@!?O=R o@*      ??!       2	lv???/??lv???/??!lv???/??:	?ЕT@?ЕT@!?ЕT@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??ޭ?tD@yj{!RA?M@