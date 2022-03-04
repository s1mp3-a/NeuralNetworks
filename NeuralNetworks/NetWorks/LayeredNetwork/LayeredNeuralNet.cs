using System;
using System.Collections.Generic;
using NeuralNetworks.LayerWorks;

namespace NeuralNetworks.NetWorks.LayeredNetwork
{
    public class LayeredNeuralNet
    {
        private readonly List<Layer> _layers = new();

        public LayeredNeuralNet(int initLayerSynapseCount, int[] layerNeuronCounts, Func<double, double> activator = null)
        {
            GenerateLayers(initLayerSynapseCount, layerNeuronCounts, activator);
        }

        private void GenerateLayers(int initLayerSynapseCount, int[] layerNeuronCounts, Func<double, double> activator)
        {
            var currentLayer = new Layer(layerNeuronCounts[0], initLayerSynapseCount, activator);
            var previousLayer = currentLayer;
            _layers.Add(currentLayer);

            for (int i = 1; i < layerNeuronCounts.Length; i++)
            {
                currentLayer = new Layer(layerNeuronCounts[i], previousLayer.NeuronCount);
                previousLayer = currentLayer;
                _layers.Add(currentLayer);
            }
        }

        public double[] GetResult(double[] initialParams)
        {
            _layers[0].SetInputs(initialParams);
            var layerResult = _layers[0].EvaluateLayer();

            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].SetInputs(layerResult);
                layerResult = _layers[i].EvaluateLayer();
            }

            return layerResult;
        }
    }
}