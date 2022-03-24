using System;
using System.Collections.Generic;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class Layer
    {
        private readonly Neuron[] _neurons;
        private readonly ILayerEvaluator _evaluator;
        internal readonly FunctionTypeTuple _layerFunctions;

        public int NeuronCount => _neurons.Length;
        public Neuron[] Neurons => _neurons;

        public Layer(int neuronCount, int neuronSynapseCount, FunctionTypeTuple аFunctionTypeTuple = null)
        {
            _neurons = new Neuron[neuronCount];
            _evaluator = neuronCount >= 384 ? LayerEvaluators.Parallel : LayerEvaluators.Sequential;
            _layerFunctions = аFunctionTypeTuple ?? NeuronFunctions.Functions.Sigmoid;
            for (int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new Neuron(neuronSynapseCount, _layerFunctions.Activator);
            }
        }

        public void SetInputs(double[] neuronInput)
        {
            foreach (var neuron in _neurons)
            {
                neuron.SetInput(neuronInput);
            }
        }

        public double[] EvaluateLayer()
        {
            return _evaluator.Evaluate(_neurons);
        }
    }
}