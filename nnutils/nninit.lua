local function msr(fanOut)
   return math.sqrt(2/fanOut/3)
end

function nn.initialize(model, method)
   if method == 'msr' then method = msr end
   for _,m in pairs(model:findModules("nn.SpatialConvolution")) do
      m:reset(method(m.nOutputPlane*m.kH*m.kW))
   end
   for _,m in pairs(model:findModules("nn.Linear")) do
      m:reset(method(m.weight:size(1)))
   end
end

