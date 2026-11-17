/** TESIS FASE 1: Sentinel-2 2020-2023, 4 bandas + pseudoetiqueta MNDWI. */
var CFG = {
 start:'2020-01-01', endExclusive:'2024-01-01', driveFolder:'Tesis_UNet_Sentinel2_Fase1',
 crs:'EPSG:32717', transform:[10,0,0,0,-10,10000000], maxSceneCloudPercent:90,
 cloudProbability:40, cloudBufferMeters:60, minClearObservations:3,
 mndwiThreshold:0.0, nodata:65535, fileDimensions:8192, exportQC:true, reductionParallelScale:4, previewImages:false
};
var INPUT_BANDS=['B2','B3','B4','B8'];
var ANALYSIS_BANDS=['B2','B3','B4','B8','B11'];
var GRID=ee.Projection(CFG.crs,CFG.transform);
var ZONES=[
 {name:'golfo_guayaquil',roi:ee.Geometry.Rectangle([-80.20,-2.82,-79.80,-2.38],'EPSG:4326',false)},
 {name:'lagos_imbabura',roi:ee.Geometry.Rectangle([-78.47,0.10,-78.02,0.48],'EPSG:4326',false)},
 {name:'rio_napo',roi:ee.Geometry.Rectangle([-77.80,-1.14,-77.42,-0.92],'EPSG:4326',false)}
];
function joinedCollection(roi){
 var sr=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
  .filterDate(CFG.start,CFG.endExclusive).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',CFG.maxSceneCloudPercent));
 var clouds=ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(roi).filterDate(CFG.start,CFG.endExclusive);
 var joined=ee.ImageCollection(ee.Join.saveFirst('cloud_probability').apply({primary:sr,secondary:clouds,
  condition:ee.Filter.equals({leftField:'system:index',rightField:'system:index'})})).filter(ee.Filter.notNull(['cloud_probability']));
 print('Escenas SR antes/despues de unir probabilidades',sr.size(),joined.size());
 return joined;
}
function prepareImage(image,roi){
 image=image.clip(roi.buffer(CFG.cloudBufferMeters+100));
 var probability=ee.Image(image.get('cloud_probability')).select('probability');
 var scl=image.select('SCL');
 var cloudOrShadow=probability.gte(CFG.cloudProbability).or(scl.eq(3)).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
 cloudOrShadow=cloudOrShadow.reproject(GRID).focalMax({radius:CFG.cloudBufferMeters,kernelType:'circle',units:'meters'});
 var clear=cloudOrShadow.not().and(scl.neq(0)).and(scl.neq(1)).and(scl.neq(11)).and(probability.mask())
  .and(image.select('B8A').mask()).and(image.select('B9').mask());
 var swir=image.select('B11').resample('bilinear').reproject(GRID);
 var optical=image.select(INPUT_BANDS).reproject(GRID).addBands(swir);
 var common=optical.mask().reduce(ee.Reducer.min()).and(optical.gte(0).reduce(ee.Reducer.min()))
  .and(optical.lt(CFG.nodata).reduce(ee.Reducer.min())).and(clear);
 return optical.updateMask(common).toFloat().copyProperties(image,['system:time_start','system:index']);
}
function buildZone(zone){
 print('Zona',zone.name,zone.roi);
 var clean=joinedCollection(zone.roi).map(function(image){return prepareImage(ee.Image(image),zone.roi);});
 var median=clean.select(ANALYSIS_BANDS).reduce(ee.Reducer.median(),CFG.reductionParallelScale).rename(ANALYSIS_BANDS).setDefaultProjection(GRID).clip(zone.roi);
 var count=clean.select('B3').reduce(ee.Reducer.count(),CFG.reductionParallelScale).rename('clear_count').setDefaultProjection(GRID).clip(zone.roi);
 var denominator=median.select('B3').add(median.select('B11'));
 var valid=median.mask().reduce(ee.Reducer.min()).and(count.gte(CFG.minClearObservations)).and(denominator.gt(0));
 var mndwi=median.expression('(green - swir) / (green + swir)',{green:median.select('B3'),swir:median.select('B11')})
  .rename('MNDWI').updateMask(valid);
 var water=mndwi.gt(CFG.mndwiThreshold).rename('water_mask');
 var product=median.select(INPUT_BANDS).round().addBands(water).updateMask(valid).clip(zone.roi)
  .unmask({value:CFG.nodata,sameFootprint:false}).toUint16();
 Map.addLayer(zone.roi,{color:'orange'},zone.name+' ROI',false);
 if(CFG.previewImages){Map.addLayer(median.updateMask(valid),{bands:['B4','B3','B2'],min:0,max:3000},zone.name+' RGB',zone.name==='golfo_guayaquil');
 Map.addLayer(water.selfMask(),{palette:['00bfff']},zone.name+' agua',false);
 Map.addLayer(mndwi,{min:-1,max:1,palette:['brown','white','blue']},zone.name+' MNDWI',false);
 Map.addLayer(count,{min:0,max:50,palette:['red','yellow','green']},zone.name+' observaciones validas',false);}
 var prefix='S2_'+zone.name+'_2020_2023';
 Export.image.toDrive({image:product,description:prefix,folder:CFG.driveFolder,fileNamePrefix:prefix,
  region:zone.roi,crs:CFG.crs,crsTransform:CFG.transform,maxPixels:1e10,shardSize:256,fileDimensions:CFG.fileDimensions,
  fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:CFG.nodata}});
 var provenance=ee.Feature(zone.roi,{
  zone:zone.name,start:CFG.start,end_exclusive:CFG.endExclusive,sr_collection:'COPERNICUS/S2_SR_HARMONIZED',
  cloud_collection:'COPERNICUS/S2_CLOUD_PROBABILITY',matched_scene_count:clean.size(),composite:'median',
  bands:'B2,B3,B4,B8,water_mask',stored_units:'SR_x_10000',mask_formula:'(B3-B11)/(B3+B11)',mndwi_threshold:CFG.mndwiThreshold,
  cloud_probability_threshold:CFG.cloudProbability,scene_cloud_percent_max:CFG.maxSceneCloudPercent,
  cloud_buffer_m:CFG.cloudBufferMeters,min_clear_observations:CFG.minClearObservations,crs:CFG.crs,
  crs_transform:CFG.transform.join(','),pixel_m:10,swir_native_pixel_m:20,swir_resampling:'bilinear',
  reduction_parallel_scale:CFG.reductionParallelScale,nodata:CFG.nodata,label_type:'MNDWI_pseudolabel'
 });
 return {provenance:provenance,count:count,prefix:prefix,roi:zone.roi};
}
var provenanceRows=[];
ZONES.forEach(function(zone){
 var result=buildZone(zone); provenanceRows.push(result.provenance);
 if(CFG.exportQC){Export.image.toDrive({image:result.count.unmask({value:CFG.nodata,sameFootprint:false}).toUint16(),
  description:result.prefix+'_QC_clear_count',folder:CFG.driveFolder,fileNamePrefix:result.prefix+'_QC_clear_count',
  region:result.roi,crs:CFG.crs,crsTransform:CFG.transform,maxPixels:1e10,shardSize:256,fileDimensions:CFG.fileDimensions,
  fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:CFG.nodata}});}
});
Export.table.toDrive({collection:ee.FeatureCollection(provenanceRows),description:'S2_UNet_Fase1_proveniencia',
 folder:CFG.driveFolder,fileNamePrefix:'S2_UNet_Fase1_proveniencia',fileFormat:'GeoJSON'});
Map.centerObject(ZONES[0].roi,9);
print('FASE 1: siete tareas (tres mosaicos, tres QC y proveniencia).');
print('NoData 65535 no significa fondo; mascara valida solo 0/1. Revisar nubes visualmente.');
