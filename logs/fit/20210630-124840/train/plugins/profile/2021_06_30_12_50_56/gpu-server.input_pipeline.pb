	?????>y@?????>y@!?????>y@	"(??7}??"(??7}??!"(??7}??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?????>y@*?Z^??c@1???d)@n@AB]¡???I????(@??Y?????ļ?rEagerKernelExecute 0*	@`??:??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator{?v??2@!VB??X@){?v??2@1VB??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchO???ʒ?!??[ð??)O???ʒ?1??[ð??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`??V?I??!?xӨ????)#??u???15߼??l??:Preprocessing2F
Iterator::Modelqt???!??Ka?B??)? :vp?1tm??š??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??????2@!-Zψ^?X@)4??`[?1%??B΁?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9!(??7}??I+Q??D@Q?þu??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*?Z^??c@*?Z^??c@!*?Z^??c@      ??!       "	???d)@n@???d)@n@!???d)@n@*      ??!       2	B]¡???B]¡???!B]¡???:	????(@??????(@??!????(@??B      ??!       J	?????ļ??????ļ?!?????ļ?R      ??!       Z	?????ļ??????ļ?!?????ļ?b      ??!       JGPUY!(??7}??b q+Q??D@y?þu??M@