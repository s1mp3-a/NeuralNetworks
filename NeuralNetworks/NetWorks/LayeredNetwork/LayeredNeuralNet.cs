using System;
using System.Collections.Generic;
using NeuralNetworks.LayerWorks;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.NetWorks.LayeredNetwork
{
    public class LayeredNeuralNet
    {
        private readonly List<Layer> _layers = new();
        internal List<Layer> Layers => _layers;
        internal double[] Output { get; private set; }

        public LayeredNeuralNet(int initLayerSynapseCount, int[] layerNeuronCounts, FunctionTypeTuple functionType = null)
        {
            GenerateLayers(initLayerSynapseCount, layerNeuronCounts, functionType);
        }

        private void GenerateLayers(int initLayerSynapseCount, int[] layerNeuronCounts, FunctionTypeTuple functionType)
        {
            var currentLayer = new Layer(layerNeuronCounts[0], initLayerSynapseCount, functionType);
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

            for (int i = 1; i < _layers.Count; i++)
            {
                _layers[i].SetInputs(layerResult);
                layerResult = _layers[i].EvaluateLayer();
            }

            Output = layerResult;
            return layerResult;
        }
    }
}