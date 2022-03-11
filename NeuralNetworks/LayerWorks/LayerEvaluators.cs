using System;

namespace NeuralNetworks.LayerWorks
{
    internal static class LayerEvaluators
    {
        // private static readonly Tuple<SequentialLayerEvaluator, ParallelLayerEvaluator> _singleton =
            // new(new SequentialLayerEvaluator(), new ParallelLayerEvaluator());
        private static Tuple<SequentialLayerEvaluator, ParallelLayerEvaluator> _singleton { get; } =
            new(new(), new());

        public static ILayerEvaluator Sequential => _singleton.Item1;
        public static ILayerEvaluator Parallel => _singleton.Item2;
    }
}