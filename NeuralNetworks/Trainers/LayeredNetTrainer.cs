using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;
using NeuralNetworks.LayerWorks;
using NeuralNetworks.NetWorks.LayeredNetwork;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.Trainers;
using Vector = Vector<double>;
public class LayeredNetTrainer
{
    private readonly LayeredNeuralNet _net;
    private readonly Func<Vector, Vector> _activator = NeuronFunctions.VectorFunc.Sigmoid;
    private readonly double[][] _examples;
    private readonly double[][] _inputs;

    public LayeredNetTrainer(LayeredNeuralNet net, (double[][] results, double[][] inputs) data)
    {
        _net = net;
        _examples = data.results;
        _inputs = data.inputs;
    }

    public void Train(int epochCount, double eta)
    {
        for (int i = 0; i < epochCount; i++)
        {
            TrainNetwork(eta, out var error);
            Console.WriteLine($"Epoch: {i}.\tDeviation: {error}");
        }
    }

    private void TrainNetwork(double eta, out double error)
    {
        error = 0d;
        for (int exampleIndex = 0; exampleIndex < _examples.GetLength(0); exampleIndex++)
        {
            var layersOuts = GetLayersOuts(_inputs[exampleIndex]);
            var weightsOfNetwork = GetWeights();

            var currentLayerResult = layersOuts.Last();
            var exampleResult = Vector.Build.DenseOfArray(_examples[exampleIndex]);

            var delta = - (exampleResult - currentLayerResult) * currentLayerResult * (1 - currentLayerResult);

            var previousLayerResult = layersOuts[^2];

            var weightsOfLayer = weightsOfNetwork[^1];

            CorrectLayerWeights(layerIndex: _net.Layers.Count - 1, weightsOfLayer, delta, eta, previousLayerResult);
        }
    }

    private void CorrectLayerWeights(int layerIndex, Vector[] weightsOfLayer, Vector delta, double eta, Vector previousLayerResult)
    {
        var layer = _net.Layers[layerIndex];
        var newWeightsOfLayer = new Vector[weightsOfLayer.Length];
        for (int i = 0; i < newWeightsOfLayer.Length; i++)
        {
            newWeightsOfLayer[i] = weightsOfLayer[i] + eta * delta * previousLayerResult;
        }
    }


    private Vector[][] GetWeights()
    {
        var weightsOfNetwork = new Vector[_net.Layers.Count][];

        for (int i = 0; i < _net.Layers.Count; i++)
        {
            var layer = _net.Layers[i];
            var layerWeights = new Vector[layer.NeuronCount];
            for (int j = 0; j < layer.NeuronCount; j++)
            {
                layerWeights[j] = Vector.Build.Dense(layer.Neurons[j]._weights.ToArray());
            }

            weightsOfNetwork[i] = layerWeights;
        }
        
        return weightsOfNetwork;
    }

    private Vector[] GetLayersOuts(double[] input)
    {
        var layerOuts = new Vector[_net.Layers.Count];
        _net.Layers[0].SetInputs(input);
        layerOuts[0] = Vector.Build.Dense(_net.Layers[0].EvaluateLayer());
        
        for (int i = 1; i < _net.Layers.Count; i++)
        {
            _net.Layers[i].SetInputs(layerOuts[i-1].AsArray());
            layerOuts[i] = Vector.Build.Dense(_net.Layers[i].EvaluateLayer());
        }

        return layerOuts;
    }
}