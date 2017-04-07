-- ============================================================================
-- MNIST dataset loader.
--
-- Credit:
-- https://github.com/torch/demos/tree/master/train-a-digit-classifier
-- ============================================================================

require 'torch'
require 'paths'

mnist = {}

local DIR = 'datasets/mnist/'
local PATH_TRAINSET = DIR .. 'train_32x32.t7'
local PATH_TESTSET = DIR .. 'test_32x32.t7'


local function loadDataset(fname, maxLoad)

    local f = torch.load(fname, 'ascii')
    local data = f.data:type(torch.getdefaulttensortype())/255.
    local labels = f.labels

    local nExample = f.data:size(1)
    if maxLoad and maxLoad > 0 and maxLoad < nExample then
        nExample = maxLoad
    end
    print('<mnist> loading ' .. nExample .. ' examples')

    data = data[{{1,nExample},{},{},{}}]
    labels = labels[{{1,nExample}}]

    local dataset = {}
    dataset.data = data
    dataset.labels = labels

    function dataset:size()
        return nExample
    end

    local labelVector = torch.zeros(10)

    setmetatable(dataset, {__index = function(self, index)
        local input = self.data[index]
        local class = self.labels[index]
        local label = labelVector:zero()
        label[class] = 1
        local example = {input, label}
        return example
    end})

    return dataset
end


function mnist.loadTrainSet(maxLoad)
    return loadDataset(PATH_TRAINSET, maxLoad)
end


function mnist.loadTestSet(maxLoad)
    return loadDataset(PATH_TESTSET, maxLoad)
end
