	???Ry@???Ry@!???Ry@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???Ry@TUh ,d@1M??E?n@A?e0F$
??I?&1????rEagerKernelExecute 0*	??~j?Q?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?)???<@!D???R?X@)?)???<@1D???R?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?"1Aߒ?!@&???D??)?"1Aߒ?1@&???D??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??˶?֠?!s?#???)? ?X4???1??T??:Preprocessing2F
Iterator::Modelo+?6??!Ţ??x??)???2#r?1??}?E??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapZ??/-?<@!?.3???X@)?!9??U`?1$g?a *|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIЖ?k?#D@Q0i>?$?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	TUh ,d@TUh ,d@!TUh ,d@      ??!       "	M??E?n@M??E?n@!M??E?n@*      ??!       2	?e0F$
???e0F$
??!?e0F$
??:	?&1?????&1????!?&1????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qЖ?k?#D@y0i>?$?M@