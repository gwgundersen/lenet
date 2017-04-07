-- ============================================================================
-- SVHN dataset loader.
-- ============================================================================


svhn = {}
--local DIR = '/tigress/gwg3/datasets/svhn/'
local DIR = 'datasets/svhn/'


local function loadDataset(fname, maxLoad)
    local data = torch.load(fname, 'ascii')
    data.X = data.X:type(torch.getdefaulttensortype())

    local nExample = data.X:size(1)
    if maxLoad and maxLoad > 0 and maxLoad < nExample then
        nExample = maxLoad
    end
    print('<svhn> loading ' .. nExample .. ' examples')

--    local X = data.X:cuda():transpose(3,4)/255.
--    local Y = data.y:cuda():squeeze()
    local X = data.X:transpose(3,4)/255.
    local Y = data.y:squeeze()

    local dataset = {}
    dataset.data = X[{{1,nExample},{},{},{}}]
    dataset.labels = Y[{{1,nExample}}]
    local labelVector = torch.zeros(10)

    setmetatable(dataset, {__index = function(self, index)
        local input = self.data[index]
        local class = self.labels[index]
        local label = labelVector:zero()
        label[class] = 1
        local example = {input, label}
        return example
    end})

    function dataset:size()
        return nExample
    end

    return dataset
end


function svhn.loadTrainSet(maxLoad)
    return loadDataset(DIR .. 'train_32x32.t7', maxLoad)
end


function svhn.loadTestSet(maxLoad)
    return loadDataset(DIR .. 'test_32x32.t7', maxLoad)
end
