using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class SequentialLayerEvaluator : ILayerEvaluator
    {
        public double[] Evaluate(Neuron[] neurons)
        {
            var res = new double[neurons.Length];
            
            for (int i = 0; i < neurons.Length; i++)
            {
                res[i] = neurons[i].Activate();
            }

            return res;
        }
    }
}