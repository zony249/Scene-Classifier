	?HZ?x@?HZ?x@!?HZ?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?HZ?x@p?4(+c@1i??QU1n@A??>Ȳ`??I?$xC???rEagerKernelExecute 0*	?K7????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??[;Q??@!??q|??X@)??[;Q??@1??q|??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??辜??!??[4??)??辜??1??[4??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??]?????!???????)x?g?ɇ?1?9p?oâ?:Preprocessing2F
Iterator::Model7???N???!f???;??)???i?:m?1sWZY???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapf?-???@!?????X@)9CqǛ?V?1?A?""r?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?H????C@Q4?}InN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p?4(+c@p?4(+c@!p?4(+c@      ??!       "	i??QU1n@i??QU1n@!i??QU1n@*      ??!       2	??>Ȳ`????>Ȳ`??!??>Ȳ`??:	?$xC????$xC???!?$xC???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?H????C@y4?}InN@