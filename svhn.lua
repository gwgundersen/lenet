-- ============================================================================
-- SVHN dataset loader.
-- ============================================================================


svhn = {}
local DIR = '/tigress/gwg3/datasets/svhn/'
--local DIR = '/Users/gwg/lenet5/svhn/'


local function loadAndProcess(fname)
    local data = torch.load(fname, 'ascii')

    local X = data.X:cuda():transpose(3,4)/255.
    local Y = data.y:cuda():squeeze()
--    local X = data.X:transpose(3,4)/255.
--    local Y = data.y:squeeze()

    local dataset = {}
    dataset.data = X
    dataset.labels = Y
    local labelvector = torch.zeros(10)

    setmetatable(dataset, {__index = function(self, index)
        local input = self.data[index]
        local class = self.labels[index]
        local label = labelvector:zero()
        label[class] = 1
        local example = {input, label}
        return example
    end})

    function dataset:size()
        return Y:nDimension()
    end

    return dataset
end


function svhn.loadTrainSet()
    return loadAndProcess(DIR .. 'train_32x32.t7')
end


function svhn.loadTestSet()
    return loadAndProcess(DIR .. 'test_32x32.t7')
end
