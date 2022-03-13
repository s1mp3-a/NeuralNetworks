using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class Layer
    {
        private readonly Neuron[] _neurons;
        private readonly ILayerEvaluator _evaluator;

        public int NeuronCount => _neurons.Length;
        public Neuron[] Neurons => _neurons;

        public Layer(int neuronCount, int neuronSynapseCount, Func<double, double> neuronActivator = null)
        {
            _neurons = new Neuron[neuronCount];
            _evaluator = neuronCount >= 384 ? LayerEvaluators.Parallel : LayerEvaluators.Sequential;
            for (int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new Neuron(neuronSynapseCount, neuronActivator);
            }
        }

        public void SetInputs(double[] neuronInput)
        {
            foreach (var neuron in _neurons)
            {
                neuron.SetInput(neuronInput);
            }
        }

        public void SetWeights(Vector<double>[] weights)
        {
            int a = 0;
            foreach (var neuron in _neurons)
            {
                neuron._weights = weights[a++];
            }
        }

        public double[] EvaluateLayer()
        {
            return _evaluator.Evaluate(_neurons);
        }
    }
}