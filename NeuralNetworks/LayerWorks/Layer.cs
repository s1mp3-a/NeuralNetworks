using System;
using System.Collections.Generic;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class Layer
    {
        private readonly List<Neuron> _neurons = new();
        private ILayerEvaluator _evaluator;

        public int NeuronCount => _neurons.Count;

        public Layer(int neuronCount, int neuronSynapseCount, Func<double, double> neuronActivator = null)
        {
            _evaluator = neuronCount >= 384 ? LayerEvaluators.Parallel : LayerEvaluators.Sequential;
            for (int i = 0; i < neuronCount; i++)
            {
                _neurons.Add(new Neuron(neuronSynapseCount, neuronActivator));
            }
        }

        public void SetInputs(double[] neuronInput)
        {
            _neurons.ForEach(neuron => neuron.SetInput(neuronInput));
        }

        public double[] EvaluateLayer()
        {
            return _evaluator.Evaluate(_neurons);
        }
    }
}