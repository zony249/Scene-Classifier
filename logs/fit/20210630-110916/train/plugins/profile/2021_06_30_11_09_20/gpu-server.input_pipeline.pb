	?|]??Uy@?|]??Uy@!?|]??Uy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?|]??Uy@??-d@1~?T??;n@A ??q???I?g?o}8 @rEagerKernelExecute 0*	?Q??G?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator ??L?<@!v6B??X@) ??L?<@1v6B??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?L???Ɣ?!?>?w????)?L???Ɣ?1?>?w????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-???!??Wۺ?)xe?????1?>O?s֡?:Preprocessing2F
Iterator::Model<??~K??!E9|l?۽?)&??|?k?1??R?(??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:?%??<@!??$??X@)??????^?1x??8?z?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI,N?q?)D@QԱ??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-d@??-d@!??-d@      ??!       "	~?T??;n@~?T??;n@!~?T??;n@*      ??!       2	 ??q??? ??q???! ??q???:	?g?o}8 @?g?o}8 @!?g?o}8 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q,N?q?)D@yԱ??M@