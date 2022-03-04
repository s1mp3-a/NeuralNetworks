using System.Collections.Generic;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class ParallelLayerEvaluator : ILayerEvaluator
    {
        public double[] Evaluate(List<Neuron> neurons)
        {
            var res = new double[neurons.Count];
            Parallel.For(0, neurons.Count, i =>
            {
                res[i] = neurons[i].Activate();
            });
            return res;
        }
    }
}