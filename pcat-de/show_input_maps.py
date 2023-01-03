import matplotlib
import matplotlib.pyplot as plt
import numpy as np


			if gdat.show_input_maps:
				plt.figure(figsize=(15, 5))
				plt.subplot(1,3,1)
				plt.title('noise realization')
				plt.imshow(noise_realization)
				plt.colorbar()
				plt.subplot(1,3,2)
				plt.title('image + gaussian noise')
				showim = image.copy()
				plt.imshow(showim)
				plt.colorbar()
				plt.subplot(1,3,3)
				plt.title('error map')
				plt.imshow(error, vmin=np.percentile(error, 5), vmax=np.percentile(error, 95))
				plt.colorbar()
				plt.show()



					if gdat.show_input_maps:
			plt.figure()
			plt.title(extname)
			plt.imshow(image, origin='lower', vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 95))
			plt.colorbar()
			plt.show()


			if show_input_maps:
				plt.figure()
				plt.imshow(noise_realization, vmin=np.nanpercentile(noise_realization, 5), vmax=np.nanpercentile(noise_realization, 95))
				plt.title('noise raelization')
				plt.colorbar()
				plt.show()


	if show_input_maps:

		plt.figure(figsize=(12, 4))
		plt.subplot(1,3,1)
		plt.title('image map')
		plt.imshow(image, vmin=np.percentile(image, 5), vmax=np.percentile(image, 95), origin='lower')
		plt.colorbar()
		plt.subplot(1,3,2)
		plt.title('err map')
		plt.imshow(error, vmin=np.percentile(error, 5), vmax=np.percentile(error, 95), origin='lower')
		plt.colorbar()
		plt.subplot(1,3,3)
		plt.title('mask')
		plt.imshow(mask, origin='lower')

		plt.show()


								if show_input_maps:
									plt.figure()
									plt.title(template_file_name)
									plt.imshow(template, origin='lower')
									plt.colorbar()
									plt.show()


											if show_input_maps:
									plt.figure()
									plt.title('loaded directly from fits extension')
									plt.imshow(template, origin='lower')
									plt.colorbar()
									plt.show()



							if show_input_maps:
								plt.figure()
								plt.title(template_name)
								plt.imshow(template, origin='lower')
								plt.colorbar()
								plt.show()



								if show_input_maps:
									plt.figure(figsize=(8, 4))
									plt.subplot(1,2,1)
									plt.title('injected amp is '+str(np.round(gdat.inject_sz_frac*temp_mock_amps_dict[gdat.band_dict[band]]*flux_density_conversion_dict[gdat.band_dict[band]], 4)))
									plt.imshow(template_inject, origin='lower', cmap='Greys')
									plt.colorbar()
									plt.subplot(1,2,2)
									plt.title('image + sz')
									plt.imshow(image, origin='lower', cmap='Greys')
									plt.colorbar()
									plt.tight_layout()
									plt.show()



						if show_input_maps:
							plt.figure()
							plt.subplot(1,2,1)
							plt.title('resized template -- '+gdat.template_order[t])
							plt.imshow(resized_template, cmap='Greys', origin='lower')
							plt.colorbar()
							if i > 0:
								plt.axhline(x_max_pivot, color='r', linestyle='solid')
								plt.axvline(y_max_pivot, color='r', linestyle='solid')
							plt.subplot(1,2,2)
							plt.title('image + '+gdat.template_order[t])
							plt.imshow(resized_image, origin='lower', cmap='Greys')
							plt.colorbar()
							if i > 0:
								plt.axhline(x_max_pivot, color='r', linestyle='solid')
								plt.axvline(y_max_pivot, color='r', linestyle='solid')
							plt.tight_layout()
							plt.show()


										if show_input_maps:
								plt.figure()
								plt.title(gdat.template_order[t]+', '+gdat.tail_name)
								# plt.title('zero-centered template -- '+gdat.template_order[t]+', '+gdat.tail_name)
								plt.imshow(resized_template, cmap='Greys', origin='lower', vmin=np.percentile(resized_template, 5), vmax=np.percentile(resized_template, 95))
								plt.colorbar()
								# plt.savefig('../zc_dust_temps/zc_dust_'+gdat.tail_name+'_band'+str(i)+'.png', bbox_inches='tight')
								plt.show()





								if show_input_maps:
									plt.figure()
									plt.subplot(1,2,1)
									plt.title('injected dust')
									plt.imshow(resized_template, origin='lower')
									plt.colorbar()
									plt.subplot(1,2,2)
									plt.title('image + dust')
									plt.imshow(resized_image, origin='lower')
									plt.colorbar()
									plt.tight_layout()
									plt.show()

							if show_input_maps:
						plt.figure()
						plt.title(diffuse_comp.shape)
						plt.imshow(diffuse_comp)
						plt.colorbar()
						plt.show()


							if show_input_maps:
						plt.figure()
						plt.title(cropped_diffuse_comp.shape)
						plt.imshow(cropped_diffuse_comp)
						plt.colorbar()
						plt.show()

							if show_input_maps:
						plt.figure()
						plt.title('resized image with cropped diffuse comp')
						plt.imshow(resized_image)
						plt.colorbar()
						plt.show()



						if show_input_maps:
				plt.figure()
				plt.title('data, '+gdat.tail_name)
				plt.imshow(self.data_array[i], vmin=np.percentile(self.data_array[i], 5), vmax=np.percentile(self.data_array[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()

				if i > 0:
					plt.axhline(x_max_pivot, color='r', linestyle='solid')
					plt.axvline(y_max_pivot, color='r', linestyle='solid')

				plt.xlim(0, self.data_array[i].shape[0])
				plt.ylim(0, self.data_array[i].shape[1])
				# if i==2:
					# print('saving this one boyyyy')
					# plt.savefig('../data_sims/'+gdat.tail_name+'_data_500micron.png', bbox_inches='tight')
				plt.show()

				plt.figure()
				plt.title('errors')
				plt.imshow(self.errors[i], vmin=np.percentile(self.errors[i], 5), vmax=np.percentile(self.errors[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()
				if i > 0:
					plt.axhline(x_max_pivot, color='r', linestyle='solid')
					plt.axvline(y_max_pivot, color='r', linestyle='solid')
				plt.xlim(0, self.errors[i].shape[0])
				plt.ylim(0, self.errors[i].shape[1])
				plt.show()







