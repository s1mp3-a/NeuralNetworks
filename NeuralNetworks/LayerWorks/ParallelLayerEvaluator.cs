using System.Collections.Generic;
using System.Threading.Tasks;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class ParallelLayerEvaluator : ILayerEvaluator
    {
        public double[] Evaluate(Neuron[] neurons)
        {
            var res = new double[neurons.Length];
            Parallel.For(0, neurons.Length, i =>
            {
                res[i] = neurons[i].Activate();
            });
            return res;
        }
    }
}